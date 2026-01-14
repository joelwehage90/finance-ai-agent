from __future__ import annotations

from typing import Any, Dict, List, Optional

from .format_spec import FormatSpec


def _q(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', "'"]):
        return f"“{s}”"
    return s


def _list(vals: Any, *, max_items: int = 6) -> str:
    if not isinstance(vals, list):
        return _q(vals)
    items = [_q(x) for x in vals[:max_items]]
    if len(vals) > max_items:
        items.append("…")
    return ", ".join(items)


def _op_sv(op: str) -> str:
    return {
        "eq": "=",
        "neq": "≠",
        "in": "i",
        "not_in": "inte i",
        "contains": "innehåller",
        "gt": ">",
        "gte": "≥",
        "lt": "<",
        "lte": "≤",
    }.get(op, op)


def filter_expr_to_sv(expr: Any) -> str:
    """
    Render filter_expr as a human-readable Swedish boolean expression.
    """
    if not isinstance(expr, dict) or not expr:
        return ""
    if "rule" in expr and isinstance(expr.get("rule"), dict):
        return filter_expr_to_sv(expr.get("rule"))

    op = str(expr.get("op") or "").lower()
    if op in {"and", "or"}:
        args = expr.get("args") if isinstance(expr.get("args"), list) else []
        parts = [filter_expr_to_sv(a) for a in args]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        joiner = " och " if op == "and" else " eller "
        if len(parts) == 1:
            return parts[0]
        return "(" + joiner.join(parts) + ")"
    if op == "not":
        inner = filter_expr_to_sv(expr.get("arg"))
        return f"inte ({inner})" if inner else ""

    # leaf rule
    if "col" in expr and "op" in expr:
        col = str(expr.get("col") or "").strip()
        rop = str(expr.get("op") or "").strip()
        val = expr.get("value")
        sym = _op_sv(rop)
        if rop in {"in", "not_in"}:
            return f"{col} {sym} ({_list(val)})"
        if rop == "contains":
            return f"{col} {sym} {_q(val)}"
        return f"{col} {sym} {_q(val)}"

    return ""


def _derive_to_sv(derive: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for d in derive:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()
        op = str(d.get("op") or "").strip()
        a = d.get("a")
        b = d.get("b")
        inputs = d.get("inputs")
        if not name or not op:
            continue
        if op == "sum" and isinstance(inputs, list):
            out.append(f"{name} = summa({_list(inputs, max_items=10)})")
        elif op == "pct_change":
            out.append(f"{name} = % förändring({_q(a)} vs {_q(b)})")
        elif op in {"sub", "add", "mul", "div", "ratio"}:
            out.append(f"{name} = {op}({_q(a)}, {_q(b)})")
        elif op in {"abs", "neg"}:
            out.append(f"{name} = {op}({_q(a)})")
        else:
            out.append(f"{name} = {op}(…)")
    return out


def build_format_summary_sv(
    *,
    spec: FormatSpec,
    payload_format: Optional[Dict[str, Any]] = None,
    derived_columns: Optional[List[Dict[str, Any]]] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Deterministic summary for UI and future frontends.
    Returns:
      {"summary_sv": str, "steps_sv": [str,...]}
    """
    steps: List[str] = []

    unit_disp = None
    if payload_format and isinstance(payload_format.get("unit"), str):
        unit_disp = payload_format.get("unit")
    else:
        unit_disp = spec.unit

    steps.append(f"Enhet: {unit_disp}.")

    # decimals + column_decimals
    if spec.column_decimals:
        # show a compact subset
        items = []
        for k, v in list(spec.column_decimals.items())[:8]:
            items.append(f"{k}={v}")
        tail = ", ".join(items) + (", …" if len(spec.column_decimals) > 8 else "")
        steps.append(f"Decimaler: standard={spec.decimals}, per kolumn: {tail}.")
    else:
        steps.append(f"Decimaler: {spec.decimals}.")

    # sorting/top_n/totals
    if spec.sort:
        s0 = spec.sort[0]
        col = s0.col or "(högerkolumn)"
        steps.append(f"Sortering: {col} {s0.dir}.")
    if spec.top_n:
        steps.append(f"Topp {spec.top_n}.")
    if spec.include_totals is False:
        steps.append("Totalsrader: döljs.")
    elif spec.include_totals is True:
        steps.append("Totalsrader: visas.")

    # rename
    renamed = None
    if payload_format and isinstance(payload_format.get("renamed_columns"), dict):
        renamed = payload_format.get("renamed_columns")
    elif spec.rename_columns:
        renamed = spec.rename_columns
    if isinstance(renamed, dict) and renamed:
        pairs = [f"{k}→{v}" for k, v in list(renamed.items())[:8]]
        tail = ", ".join(pairs) + (", …" if len(renamed) > 8 else "")
        steps.append(f"Kolumnnamn: {tail}.")

    # derive
    dcols = derived_columns
    if dcols is None:
        dcols = [d.model_dump(mode="json") for d in (spec.derive or [])]
    if dcols:
        for line in _derive_to_sv(dcols):
            steps.append(f"Beräkning: {line}.")

    # filters (prefer filter_expr)
    if spec.filter_expr:
        txt = filter_expr_to_sv(spec.filter_expr)
        if txt:
            steps.append(f"Filter: {txt}.")
    elif spec.filter_groups:
        # basic fallback text
        parts = []
        for g in spec.filter_groups[:5]:
            op = g.op
            joiner = " och " if op == "and" else " eller "
            rp = []
            for r in g.rules[:10]:
                rp.append(f"{r.col} {_op_sv(r.op)} {_q(r.value)}")
            if rp:
                parts.append("(" + joiner.join(rp) + ")")
        if parts:
            steps.append("Filter: " + " och ".join(parts) + ".")
    elif spec.filters:
        rp = [f"{r.col} {_op_sv(r.op)} {_q(r.value)}" for r in spec.filters[:10]]
        if rp:
            steps.append("Filter: " + " och ".join(rp) + ".")

    # If there are any explicit notes, keep summary lean but mention.
    if notes:
        steps.append("Noteringar finns.")

    # Make a compact one-liner summary as well
    summary = " ".join(steps)
    return {"summary_sv": summary, "steps_sv": steps}


