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

    header = "| Name | NID | CaseA (Prev Month → Current Month → Forecast) | CaseB (Prev Month → Current Month → Forecast) | Gap (h) |\n"
    header += "|------|-----|---------------------|---------------------|---------|\n"
    lines = []
    for r in table_rows:
        lines.append(
            f"| {r['name']} | {r['nid']} | "
            f"{r['prev_caseA']}→**{r['caseA']}**→{r['forecast_caseA']} | "
            f"{r['prev_caseB']}→**{r['caseB']}**→{r['forecast_caseB']} | "
            f"{r.get('gap_h', 0.0):.2f} |"
        )
    return header + "\n".join(lines)

# ----------------------- LLM Forecast -----------------------
# ----------------------- Unified LLM Forecast + Summary -----------------------
def _maybe_llm_forecast_and_summary(payload: Payload, risky_sorted: List[Employee]) -> Optional[Dict[str, Any]]:
    """
    Ask gpt-4.1 to generate BOTH:
      - Forecast table (with prev, cur, forecast CaseA/B)
      - Bullet-point summary

    Returns dict with {"table_rows": [...], "summary": "..."} or None on failure.
    """
    import time

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("⚠️ No OPENAI_API_KEY found; skipping LLM.")
        return None

    top_n = int(os.getenv("LLM_FORECAST_TOPN", "20"))
    focus = risky_sorted[:top_n]
    log(f"🧠 Running unified LLM forecast+summary for top {len(focus)} risky employees…")

    factor_hint = _default_forecast_factor(payload.meta)
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

    system = (
        "You are a data analyst. Your job:\n"
        "1. Forecast month-end CaseA and CaseB for each employee.\n"
        "2. Write a short bullet-point summary of the forecasts.\n\n"
        "IMPORTANT:\n"
        "- Output STRICT JSON only, no prose outside JSON.\n"
        "- Schema must exactly match:\n"
        "{\n"
        '  "items": [\n'
        '    {"name": str, "nid": str, "caseA": int, "prev_caseA": int, "forecast_caseA": int,\n'
        '     "caseB": int, "prev_caseB": int, "forecast_caseB": int, "gap_h": float}\n'
        "  ],\n"
        '  "summary": str  # markdown bullet points, ~120 words, no employee name list\n'
        "}\n"
    )
    user = f"INPUT:\n{_json.dumps(content, ensure_ascii=False)}"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        def _call_once() -> str:
            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content if (resp and resp.choices) else ""

        raw = _call_once()
        if not raw.strip():
            log("⚠️ Unified LLM: empty response; retrying once…")
            time.sleep(1)
            raw = _call_once()

        if not raw.strip():
            log("❌ Unified LLM failed: empty response.")
            return None

        parsed = _json.loads(raw)
        items = parsed.get("items", [])
        summary = parsed.get("summary", "").strip()
        log(f"✅ Unified LLM returned {len(items)} items + summary length {len(summary)}.")

        return {"items": items, "summary": summary}

    except Exception as e:
        log(f"❌ Unified LLM forecast+summary failed: {e}")
        return None


# ----------------------- Public API -----------------------
def build_insights(payload: Payload) -> Dict[str, Any]:
    log("📥 Building insights...")
    risky = [e for e in payload.employees if (e.caseA >= 5 or e.caseB >= 5)]
    log(f"⚠️ Found {len(risky)} risky employees.")

    total_gap_risky = round(sum(e.sum_gap for e in risky), 2)

    # Try unified GPT-4.1 call
    llm_result = _maybe_llm_forecast_and_summary(payload, risky)

    if llm_result:
        table_rows = llm_result["items"]
        summary_md = llm_result["summary"] or None
        used_llm_forecast = True
        used_llm_summary = True
    else:
        # fallback to rule-based
        table_rows = _rule_based_forecast_rows(risky, payload.meta)
        summary_md = None
        used_llm_forecast = False
        used_llm_summary = False

    forecast_table_md = build_forecast_table(table_rows)

    notes = (
        f"Project: {payload.meta.project_name or '-'} • Period: {payload.meta.month} "
        f"(vs {payload.meta.prev_month or 'previous month'}) • "
        f"Risky employees: {len(risky)} • Total gap (risky): {total_gap_risky}h."
    )

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
        "used_llm_forecast": used_llm_forecast,
        "used_llm_summary": used_llm_summary,
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








