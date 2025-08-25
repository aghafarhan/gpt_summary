# summarize_ai_risk.py  (library only; no FastAPI app here)
import os
from typing import List, Optional, Dict, Any
from datetime import date
from calendar import monthrange
from pydantic import BaseModel, Field

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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    top_n = int(os.getenv("LLM_FORECAST_TOPN", "20"))
    emps = [{
        "nid": e.nid,
        "name": e.name or e.nid,
        "caseA": e.caseA,
        "caseB": e.caseB,
        "prev_caseA": e.prev_caseA or 0,
        "prev_caseB": e.prev_caseB or 0,
        "sum_gap": round(e.sum_gap, 2),
    } for e in risky_sorted[:top_n]]

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
        "employees": emps
    }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Default to gpt-5-mini, but allow override via env
        model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        system = (
            "You are a data analyst. Forecast month-end CaseA and CaseB for each employee.\n"
            "OUTPUT: STRICT JSON per schema. Do not include prose."
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
            "- CaseA = days with actual<9h but posted≈10h (per employee, this month).\n"
            "- CaseB = one‑punch days (only IN or only OUT) with posted≈10h (per employee, this month).\n"
            "Rules:\n"
            "- Use ONLY provided employees (no extras).\n"
            "- Start from naive_factor * current; adjust vs previous month.\n"
            "- forecast_* are integers; NEVER below current values; cap extreme growth if unrealistic.\n"
            "- Preserve input ordering for the same employees; return items only for provided NIDs.\n\n"
            f"INPUT:\n{content}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        import json as _json
        parsed = _json.loads(resp.choices[0].message.content)
        items = parsed.get("items", [])
        rows = []
        for it in items:
            rows.append({
                "name": str(it.get("name") or ""),
                "nid": str(it.get("nid") or ""),
                "caseA": int(it.get("caseA", 0)),
                "prev_caseA": int(it.get("prev_caseA", 0)),
                "forecast_caseA": max(int(it.get("forecast_caseA", 0)), int(it.get("caseA", 0))),
                "caseB": int(it.get("caseB", 0)),
                "prev_caseB": int(it.get("prev_caseB", 0)),
                "forecast_caseB": max(int(it.get("forecast_caseB", 0)), int(it.get("caseB", 0))),
            })
        # Keep original order of risky_sorted
        nid_to_row = {r["nid"]: r for r in rows}
        ordered = []
        for e in risky_sorted[:top_n]:
            r = nid_to_row.get(e.nid)
            if r:
                r["gap_h"] = round(e.sum_gap, 2)
                ordered.append(r)
        return ordered or None
    except Exception:
        return None

# ----------------------- LLM Forecast Summary -----------------------
def maybe_llm_forecast_summary(meta: Meta, table_rows: List[Dict[str, Any]]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
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

        prompt = (
            "You are a compliance analyst. Write a short Markdown summary (~100 words) "
            "of attendance risk forecasts. Use only provided employees.\n\n"
            "Include:\n"
            "- Total risky employees\n"
            "- Notable increases vs previous month\n"
            "- Top 2–3 to watch with numeric CaseA/CaseB forecast\n"
            "- 2 action points (investigate logs, supervisor review)\n\n"
            f"Project: {meta.project_name or '-'}\n"
            f"Month: {meta.month} vs {meta.prev_month or 'prev'}\n"
            f"Rows: {compact}"
        )

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def fallback_forecast_summary(meta: Meta, table_rows: List[Dict[str, Any]]) -> str:
    risky_count = len(table_rows)
    ranked = sorted(table_rows, key=lambda r: (r["forecast_caseA"] + r["forecast_caseB"], r.get("gap_h", 0.0)), reverse=True)
    top3 = [r["name"] for r in ranked[:3]]
    parts = [
        f"**{meta.project_name or '-'} – {meta.month} (vs {meta.prev_month or 'prev'})**",
        f"Risky employees: **{risky_count}**.",
    ]
    if top3:
        parts.append("Watchlist: " + ", ".join(top3) + ".")
    parts.append("Actions: - Audit top forecasts; - Supervisor review when A or B ≥ 5; - Enable daily one‑punch alerts.")
    return "\n".join(parts)

# ----------------------- Public API (for backend.py to call) -----------------------
def build_insights(payload: Payload) -> Dict[str, Any]:
    emps = payload.employees
    meta = payload.meta

    risky = [e for e in emps if (e.caseA >= 5 or e.caseB >= 5)]
    risky_sorted = sorted(risky, key=_score_for_sort, reverse=True)
    total_gap_risky = round(sum(e.sum_gap for e in risky), 2)

    # Forecast rows
    llm_rows = _maybe_llm_forecast(payload, risky_sorted)
    table_rows = llm_rows if llm_rows is not None else _rule_based_forecast_rows(risky_sorted, meta)

    # Build forecast table
    forecast_table_md = build_forecast_table(table_rows)

    # Notes
    notes = (
        f"Project: {meta.project_name or '-'} • Period: {meta.month} (vs {meta.prev_month or 'previous month'}) • "
        f"Risky employees: {len(risky_sorted)} • Total gap (risky): {total_gap_risky}h."
    )

    # Top risky (for JSON consumers)
    top_risky_strings = [
        f"{r['name']} — Gap {r.get('gap_h', 0):.2f}h (A {r['caseA']}→{r['forecast_caseA']}, "
        f"B {r['caseB']}→{r['forecast_caseB']})"
        for r in table_rows[:10]
    ]

    # Textual summary
    summary_md = maybe_llm_forecast_summary(meta, table_rows) or fallback_forecast_summary(meta, table_rows)

    return {
        "risk_notes": notes,
        "forecast_table_md": forecast_table_md,   # ⬅️ new Markdown table
        "forecast_summary_md": summary_md,        # ⬅️ textual summary
        "top_risky": top_risky_strings,
        "risky_compare": table_rows,
        "risky_count": len(risky_sorted),
        "total_gap_risky": total_gap_risky,
        "used_llm_forecast": llm_rows is not None,
        "used_llm_summary": summary_md is not None,
    }




