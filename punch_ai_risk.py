# punch_ai_risk.py
"""
Library for forecasting and summarizing punch hours AI risk.
Can be imported into FastAPI backend, or run locally in test mode.
"""

import os
import json as _json
from typing import List, Optional, Dict, Any
from datetime import date
from calendar import monthrange
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────────────────────────

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ----------------------- Models -----------------------
class Employee(BaseModel):
    nid: str
    name: Optional[str] = ""
    caseA: int = 0
    caseB: int = 0
    sum_actual: float = 0.0
    sum_posted: float = 0.0
    sum_gap: float = 0.0
    prev_caseA: int = 0
    prev_caseB: int = 0
    prev_sum_actual: float = 0.0
    prev_sum_posted: float = 0.0
    prev_sum_gap: float = 0.0
    refs: List[str] = Field(default_factory=list)
    risk: Optional[int] = None
    delta: Optional[int] = None
    one_punch: Optional[int] = None
    prev_one_punch: Optional[int] = None

class Meta(BaseModel):
    docname: Optional[str] = None
    project_name: Optional[str] = None
    month: str
    prev_month: Optional[str] = None
    generated_at: Optional[str] = None
    posted_target: Optional[float] = None
    posted_tolerance: Optional[float] = None
    sum_only_eligible: Optional[bool] = None
    month_days_total: Optional[int] = None
    days_covered: Optional[int] = None

class Payload(BaseModel):
    meta: Meta
    employees: List[Employee]

# ----------------------- Utils -----------------------
def _month_len_from_yyyy_mm(yyyy_mm: str) -> int:
    try:
        y, m = yyyy_mm.split("-")
        return monthrange(int(y), int(m))[1]
    except Exception:
        return 30

def _default_forecast_factor(meta: Meta) -> float:
    total = meta.month_days_total or _month_len_from_yyyy_mm(meta.month)
    covered = meta.days_covered
    if not covered:
        try:
            y, m = map(int, meta.month.split("-"))
            today = date.today()
            covered = today.day if (today.year == y and today.month == m) else min(15, total - 1)
        except Exception:
            covered = 15
    covered = max(1, covered)
    return max(1.0, float(total) / float(covered))

def _score_for_sort(e: Employee) -> tuple:
    approx_risk = e.risk if e.risk is not None else (2 * e.caseA + 1 * e.caseB)
    return (approx_risk, e.caseA, e.caseB, e.sum_gap)

# ----------------------- Rule-based (fallback) -----------------------
def _rule_based_forecast_rows(risky_sorted: List[Employee], meta: Meta) -> List[Dict[str, Any]]:
    log("⚙️ Using rule-based forecast (LLM not available).")
    factor = _default_forecast_factor(meta)
    rows = []
    for e in risky_sorted:
        fA = int(round(max(0, e.caseA) * factor))
        fB = int(round(max(0, e.caseB) * factor))
        rows.append({
            "name": e.name or e.nid,
            "nid": e.nid,
            "caseA": e.caseA,
            "prev_caseA": e.prev_caseA or 0,
            "forecast_caseA": fA,
            "caseB": e.caseB,
            "prev_caseB": e.prev_caseB or 0,
            "forecast_caseB": fB,
            "gap_h": round(e.sum_gap, 2),
        })
    return rows

def build_forecast_table(table_rows: List[Dict[str, Any]]) -> str:
    """Return forecast rows as Markdown table."""
    if not table_rows:
        return "_No risky employees to forecast._"

    header = "| Name | NID | CaseA (Prev→Cur→Fc) | CaseB (Prev→Cur→Fc) | Gap (h) |\n"
    header += "|------|-----|---------------------|---------------------|---------|\n"
    lines = []
    for r in table_rows:
        lines.append(
            f"| {r['name']} | {r['nid']} | "
            f"{r['prev_caseA']}→{r['caseA']}→{r['forecast_caseA']} | "
            f"{r['prev_caseB']}→{r['caseB']}→{r['forecast_caseB']} | "
            f"{r.get('gap_h', 0.0):.2f} |"
        )
    return header + "\n".join(lines)

# ----------------------- LLM Forecast -----------------------
def _maybe_llm_forecast(payload: Payload, risky_sorted: List[Employee]) -> Optional[List[Dict[str, Any]]]:
    """Ask gpt-5-mini to forecast month-end CaseA/CaseB for risky employees.
    Returns ordered rows (matching risky_sorted order) or None on failure.
    """
    import json as _json

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("⚠️ No OPENAI_API_KEY found; skipping LLM forecast.")
        return None

    # Scope / size hints
    top_n = int(os.getenv("LLM_FORECAST_TOPN", "20"))
    focus = risky_sorted[:top_n]
    log(f"🧠 Running LLM forecast for top {len(focus)} risky employees…")

    # Naive factor for the model (days covered → month-end)
    factor_hint = _default_forecast_factor(payload.meta)

    # Compact, deterministic input object
    content = {
        "meta": {
            "project": payload.meta.project_name or "-",
            "month": payload.meta.month,
            "prev_month": payload.meta.prev_month or "",
            "month_days_total": payload.meta.month_days_total or _month_len_from_yyyy_mm(payload.meta.month),
            "days_covered": payload.meta.days_covered or None,
            "naive_forecast_factor": factor_hint
        },
        "employees": [{
            "nid": e.nid,
            "name": e.name or e.nid,
            "caseA": int(e.caseA),
            "caseB": int(e.caseB),
            "prev_caseA": int(e.prev_caseA or 0),
            "prev_caseB": int(e.prev_caseB or 0),
            "sum_gap": round(float(e.sum_gap), 2),
        } for e in focus]
    }

    # Build the strict JSON prompt
    system = (
        "You are a data analyst. Forecast month‑end CaseA and CaseB for each employee.\n"
        "OUTPUT: STRICT JSON that matches the schema. No prose."
    )
    user = (
        "Schema:\n"
        "{\n"
        '  "items": [\n'
        '    {"name": str, "nid": str, "caseA": int, "prev_caseA": int, "forecast_caseA": int,\n'
        '     "caseB": int, "prev_caseB": int, "forecast_caseB": int}\n'
        "  ]\n"
        "}\n\n"
        "Definitions:\n"
        "- caseA = count of days actual<9h while posted≈10h (this month).\n"
        "- caseB = count of one‑punch days (only IN or only OUT) with posted≈10h (this month).\n\n"
        "Rules:\n"
        "- Use ONLY the provided employees; do not invent extra rows.\n"
        "- Start from naive_factor * current; compare with previous month.\n"
        "- forecast_* must be integers and NEVER below the current values.\n"
        "- Cap extreme growth to reasonable bounds.\n"
        "- Preserve the input list semantics (we will reorder by NID on our side).\n\n"
        f"INPUT:\n{content}"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Call gpt-5-mini with JSON mode
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            # This model uses `max_completion_tokens` (not `max_tokens`)
            max_completion_tokens=800,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content if resp and resp.choices else ""
        if not raw or not raw.strip():
            log("❌ LLM forecast failed: empty response body from model.")
            return None

        # Try to parse JSON; if it fails, log a short preview
        try:
            parsed = _json.loads(raw)
        except Exception as je:
            preview = raw[:400].replace("\n", " ")
            log(f"❌ LLM forecast JSON parse error: {je}; preview: {preview}")
            return None

        items = parsed.get("items", [])
        log(f"✅ LLM forecast returned {len(items)} item(s).")

        # Map by NID, then restore the original risky_sorted order
        by_nid = {}
        for it in items:
            try:
                row = {
                    "name": str(it.get("name") or ""),
                    "nid": str(it.get("nid") or ""),
                    "caseA": int(it.get("caseA", 0)),
                    "prev_caseA": int(it.get("prev_caseA", 0)),
                    "forecast_caseA": max(int(it.get("forecast_caseA", 0)), int(it.get("caseA", 0))),
                    "caseB": int(it.get("caseB", 0)),
                    "prev_caseB": int(it.get("prev_caseB", 0)),
                    "forecast_caseB": max(int(it.get("forecast_caseB", 0)), int(it.get("caseB", 0))),
                }
                by_nid[row["nid"]] = row
            except Exception as row_err:
                log(f"⚠️ Skipping one malformed item from LLM: {row_err}")

        ordered: List[Dict[str, Any]] = []
        for e in focus:
            r = by_nid.get(e.nid)
            if r:
                r["gap_h"] = round(float(e.sum_gap), 2)
                ordered.append(r)

        if not ordered:
            log("⚠️ LLM forecast produced no matching NIDs after ordering; using fallback.")
            return None

        return ordered

    except Exception as e:
        # Network/timeouts/HTTP error handling path
        log(f"❌ LLM forecast failed: {e}")
        return None

# ----------------------- LLM Forecast Summary -----------------------
def maybe_llm_forecast_summary(meta: Meta, table_rows: List[Dict[str, Any]]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("⚠️ No OPENAI_API_KEY found; skipping LLM summary.")
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        compact = [{
            "name": r["name"], "nid": r["nid"],
            "A_prev": r["prev_caseA"], "A_cur": r["caseA"], "A_fc": r["forecast_caseA"],
            "B_prev": r["prev_caseB"], "B_cur": r["caseB"], "B_fc": r["forecast_caseB"],
            "gap_h": r.get("gap_h", 0.0)
        } for r in table_rows[:10]]

        log("📝 Asking GPT-5-mini to generate forecast summary...")
        prompt = (
            "Write a concise Markdown summary (~120 words) of attendance risk forecasts. "
            "Format the output as clear bullet points. "
            "Cover:\n"
            "- Total risky employees\n"
            "- Notable forecast changes vs previous month\n"
            "- 2–3 recommended actions\n\n"
            "Do NOT include a watchlist of names (already in table)."
            f"\nProject: {meta.project_name or '-'}\n"
            f"Month: {meta.month} vs {meta.prev_month or 'prev'}\n"
            f"Rows: {compact}"
        )
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=220,
        )
        log("✅ LLM summary generated.")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log(f"❌ LLM summary failed: {e}")
        return None

# ----------------------- Public API -----------------------
def build_insights(payload: Payload) -> Dict[str, Any]:
    log("📥 Building insights...")
    risky = [e for e in payload.employees if (e.caseA >= 5 or e.caseB >= 5)]
    log(f"⚠️ Found {len(risky)} risky employees.")

    total_gap_risky = round(sum(e.sum_gap for e in risky), 2)

    # Forecast
    llm_rows = _maybe_llm_forecast(payload, risky)
    table_rows = llm_rows if llm_rows is not None else _rule_based_forecast_rows(risky, payload.meta)
    forecast_table_md = build_forecast_table(table_rows)

    # Notes
    notes = (
        f"Project: {payload.meta.project_name or '-'} • Period: {payload.meta.month} (vs {payload.meta.prev_month or 'previous month'}) • "
        f"Risky employees: {len(risky)} • Total gap (risky): {total_gap_risky}h."
    )

    # Summary
    summary_md = maybe_llm_forecast_summary(payload.meta, table_rows)
    if not summary_md:
        log("⚠️ Falling back to heuristic summary.")
        summary_md = f"- Risky employees: {len(risky)}\n- Total risky gap: {total_gap_risky}h"

    log("✅ Insights built.")
    return {
        "risk_notes": notes,
        "forecast_table_md": forecast_table_md,
        "forecast_summary_md": summary_md,
        "risky_compare": table_rows,
        "risky_count": len(risky),
        "total_gap_risky": total_gap_risky,
        "used_llm_forecast": llm_rows is not None,
        "used_llm_summary": summary_md is not None,
    }

# ----------------------- Local Test -----------------------
if __name__ == "__main__":
    dummy = Payload(
        meta=Meta(project_name="Test Project", month="2025-08", prev_month="2025-07"),
        employees=[
            Employee(nid="123", name="Ali", caseA=6, caseB=2, sum_gap=45),
            Employee(nid="456", name="Ahmed", caseA=3, caseB=7, sum_gap=38.5),
        ]
    )
    out = build_insights(dummy)
    log("OUTPUT:", out)




