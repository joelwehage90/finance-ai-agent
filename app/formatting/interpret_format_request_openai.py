from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..settings import settings
from ..supabase_client import supabase
from .format_spec import FormatSpec, merge_with_default


def _res_data(res: Any) -> List[Dict[str, Any]]:
    data = getattr(res, "data", None)
    if data is None:
        data = res.data
    return data or []


def _get_openai_cfg() -> Tuple[str, str]:
    """
    OpenAI-compatible Chat Completions config.
    - OPENAI_BASE_URL: optional; defaults to OpenAI
    - OPENAI_API_KEY: required
    - OPENAI_MODEL: optional; defaults to a small/cheap model
    """
    base_url = (os.environ.get("OPENAI_BASE_URL") or settings.OPENAI_BASE_URL or "https://api.openai.com").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return base_url, api_key


def _openai_model() -> str:
    return os.environ.get("OPENAI_MODEL") or settings.OPENAI_MODEL or "gpt-4o-mini"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _detect_unsupported_requests(format_request: str) -> List[str]:
    """
    v1 guardrails: detect requests we do NOT support and surface as notes.
    We keep this light-weight and language-agnostic-ish (Swedish + English).
    """
    fr = (format_request or "").lower()
    notes: List[str] = []

    # Column removal / reordering (still not supported in current formatter)
    # NOTE: Adding columns IS supported via `derive`, so we should not warn on "lägg till kolumn"/"add column".
    if any(
        k in fr
        for k in [
            "ta bort kolumn",
            "remove column",
            "drop column",
            "reorder",
            "flytta kolumn",
            "byt plats på kolumn",
            "kolumnordning",
        ]
    ):
        notes.append("Kolumn-manipulation (ta bort/flytta/ändra ordning) stöds inte just nu.")

    return notes


_RE_RENAME_SV = re.compile(
    r"""(?ix)
    \b(?:döp|döpa|omdöp|namnge\s+om|byt\s+namn(?:\s+på)?)\b
    \s+\bkolumn\b
    \s+(?P<col>[A-Za-z0-9_\-]+)
    \s+\b(?:till|som)\b
    \s*(?P<name>"[^"]+"|'[^']+'|.+)$
    """
)
_RE_RENAME_EN = re.compile(
    r"""(?ix)
    \brename\b\s+\bcolumn\b
    \s+(?P<col>[A-Za-z0-9_\-]+)
    \s+\bto\b
    \s*(?P<name>"[^"]+"|'[^']+'|.+)$
    """
)

_RE_DIFF_SV = re.compile(
    r"""(?ix)
    \b(?:skillnad(?:en)?|diff(?:erens)?|delta)\b
    .*?
    \b(?:mellan|between)\b
    \s*(?P<a>[A-Za-z0-9_\-]+)
    \?\s*(?:och|and)\s*
    (?P<b>[A-Za-z0-9_\-]+)
    """
)

_RE_RATIO_SV = re.compile(
    r"""(?ix)
    \b(?:kvot|ratio|andel)\b
    .*?
    \b(?:mellan|between)\b
    \s*(?P<a>[A-Za-z0-9_\-]+)
    \?\s*(?:och|and)\s*
    (?P<b>[A-Za-z0-9_\-]+)
    """
)

_RE_DIV_SV = re.compile(
    r"""(?ix)
    \b(?P<a>[A-Za-z0-9_\-]+)\b
    \s*(?:/|delat\s+med|dividerat\s+med)\s*
    \b(?P<b>[A-Za-z0-9_\-]+)\b
    """
)

_RE_ABS_SV = re.compile(r"(?i)\babs\((?P<a>[A-Za-z0-9_\-]+)\)|\babsolut(?:värde)?\b.*?\b(?P<a2>[A-Za-z0-9_\-]+)\b")
_RE_NEG_SV = re.compile(r"(?i)\bneg\((?P<a>[A-Za-z0-9_\-]+)\)|\bminus\s+(?P<a2>[A-Za-z0-9_\-]+)\b")

_RE_RESET = re.compile(r"(?i)\b(nollställ|återställ|reset|default|standard|som vanligt|ta bort formatering)\b")
_RE_PCT_DECIMALS_SV = re.compile(r"(?i)\bprocent\b.*?\b(\d)\s*decimal")
_RE_COL_DEC_EXACT_SV = re.compile(r"(?i)\bkolumn(?:en)?\s+(?P<col>[A-Za-z0-9_\-]+)\b.*?\b(?P<d>[0-3])\s*decimal")
_RE_COL_DEC_LIST_SV = re.compile(r"(?i)\b(?P<col>[A-Za-z0-9_\-]+)\s*:\s*(?P<d>[0-3])\s*decimal")
_RE_VALUE_DECIMALS_SV = re.compile(r"(?i)\b(periodkolumner|värdekolumner)\b.*?\b(?P<d>[0-3])\s*decimal")

_RE_FILTER_EQ_SV = re.compile(r"""(?ix)
\b(?:visa\s+bara|endast|bara|filtrera|filter)\b.*?
(?P<col>[A-Za-z0-9_\-]+)\s*=\s*(?P<val>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+)
""")
_RE_FILTER_EXCL_SV = re.compile(r"""(?ix)
\b(?:exkludera|ta\s+bort)\b.*?
(?P<col>[A-Za-z0-9_\-]+)\s*=\s*(?P<val>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+)
""")
_RE_FILTER_CONTAINS_SV = re.compile(r"""(?ix)
\b(?:innehåller|contains)\b.*?
(?P<col>[A-Za-z0-9_\-]+)\s*:\s*(?P<val>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+)
""")
_RE_FILTER_NUM_SV = re.compile(r"""(?ix)
(?P<col>[A-Za-z0-9_\-]+)\s*(?P<op>>=|<=|>|<)\s*(?P<num>-?\d+(?:\.\d+)?)
""")

_RE_FILTER_OR_EQ_SV = re.compile(
    r"""(?ix)
    (?P<col>[A-Za-z0-9_\-]+)\s*=\s*(?P<a>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+)
    \s*(?:eller|or)\s*
    (?P=col)\s*=\s*(?P<b>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+)
    """
)

_RE_FILTER_OR_TWO_EQ_SV = re.compile(
    r"""(?ix)
    (?P<col1>[A-Za-z0-9_\-]+)\s*=\s*(?P<a>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+)
    \s*(?:eller|or)\s*
    (?P<col2>[A-Za-z0-9_\-]+)\s*=\s*(?P<b>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+)
    """
)

_RE_SUM_SV = re.compile(
    r"""(?ix)
    \b(?:summa|summera|sum)\b
    .*?
    \b(?:av|of)\b
    \s*(?P<a>[A-Za-z0-9_\-]+)
    \s*(?:,|\s+och\s+|\s+and\s+)\s*
    (?P<b>[A-Za-z0-9_\-]+)
    (?:\s*(?:,|\s+och\s+|\s+and\s+)\s*(?P<c>[A-Za-z0-9_\-]+))?
    (?:\s*(?:,|\s+och\s+|\s+and\s+)\s*(?P<d>[A-Za-z0-9_\-]+))?
    """
)



def _extract_rename_columns_from_text(format_request: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Deterministic helper for rename intents like:
    - 'döp kolumn rr_level_1 till Resultaträkning'
    - 'rename column rr_level_1 to \"Resultaträkning\"'
    Returns (rename_map, notes)
    """
    fr = (format_request or "").strip()
    if not fr:
        return {}, []

    m = _RE_RENAME_SV.search(fr) or _RE_RENAME_EN.search(fr)
    if not m:
        return {}, []

    col = (m.group("col") or "").strip()
    name = (m.group("name") or "").strip()
    if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
        name = name[1:-1].strip()

    if not col or not name:
        return {}, ["Kunde inte tolka kolumn-omdöpning (saknar kolumnnamn eller nytt namn)."]

    return {col: name}, []


def _extract_derive_from_text(format_request: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Deterministic helper for simple derive intents like:
      - 'lägg till kolumn som skillnaden mellan 2025-02 och 2025-01'
    Returns (derive_list, notes).
    """
    fr = (format_request or "").strip()
    if not fr:
        return [], []

    m = _RE_DIFF_SV.search(fr)
    if m:
        a = (m.group("a") or "").strip()
        b = (m.group("b") or "").strip()
        if not a or not b:
            return [], ["Kunde inte tolka diff-kolumn (saknar två kolumnnamn)."]
        return [{"name": "diff", "op": "sub", "a": a, "b": b}], []

    m2 = _RE_RATIO_SV.search(fr)
    if m2:
        a = (m2.group("a") or "").strip()
        b = (m2.group("b") or "").strip()
        if not a or not b:
            return [], ["Kunde inte tolka kvot/ratio-kolumn (saknar två kolumnnamn)."]
        return [{"name": "ratio", "op": "ratio", "a": a, "b": b}], []

    m3 = _RE_DIV_SV.search(fr)
    if m3:
        a = (m3.group("a") or "").strip()
        b = (m3.group("b") or "").strip()
        if not a or not b:
            return [], ["Kunde inte tolka division (saknar två kolumnnamn)."]
        return [{"name": "ratio", "op": "div", "a": a, "b": b}], []

    m4 = _RE_ABS_SV.search(fr)
    if m4:
        a = (m4.group("a") or m4.group("a2") or "").strip()
        if not a:
            return [], ["Kunde inte tolka abs (saknar kolumnnamn)."]
        return [{"name": f"abs_{a}", "op": "abs", "a": a}], []

    m5 = _RE_NEG_SV.search(fr)
    if m5:
        a = (m5.group("a") or m5.group("a2") or "").strip()
        if not a:
            return [], ["Kunde inte tolka neg (saknar kolumnnamn)."]
        return [{"name": f"neg_{a}", "op": "neg", "a": a}], []

    m6 = _RE_SUM_SV.search(fr)
    if m6:
        cols = [m6.group("a"), m6.group("b"), m6.group("c"), m6.group("d")]
        inputs = [str(x).strip() for x in cols if x and str(x).strip()]
        if len(inputs) < 2:
            return [], ["Kunde inte tolka sum (saknar minst två kolumnnamn)."]
        return [{"name": "sum", "op": "sum", "inputs": inputs[:10]}], []

    return [], []


def _build_context_from_tool_run(*, source_tool_run_id: str, max_cols: int = 80) -> Dict[str, Any]:
    """
    Build a small, UI/tool-agnostic context from tool_runs.response_json.
    Avoid sending raw rows to the LLM (cost + privacy); columns + meta is enough for v1.
    """
    res = (
        supabase
        .table("tool_runs")
        .select("id,tool_name,response_json")
        .eq("id", source_tool_run_id)
        .limit(1)
        .execute()
    )
    rows = _res_data(res)
    if not rows:
        raise RuntimeError(f"tool_run not found: {source_tool_run_id}")
    tr = rows[0]

    rj = tr.get("response_json") or {}
    if not isinstance(rj, dict):
        rj = {}

    cols = rj.get("columns") if isinstance(rj.get("columns"), list) else None
    table_rows = rj.get("table") if isinstance(rj.get("table"), list) else []
    meta = rj.get("meta") if isinstance(rj.get("meta"), dict) else {}

    if cols is None:
        if table_rows and isinstance(table_rows[0], dict):
            cols = list(table_rows[0].keys())
        else:
            cols = []

    cols = [str(c) for c in cols if c is not None]
    if len(cols) > max_cols:
        cols = cols[:max_cols]

    rightmost_col = cols[-1] if cols else None
    dims = meta.get("rows") if isinstance(meta.get("rows"), list) else None
    dims = [str(x) for x in dims if isinstance(x, str)] if dims else []

    totals_marker = "__TOTAL__"
    if isinstance(meta, dict) and meta.get("totals_marker"):
        totals_marker = str(meta.get("totals_marker"))

    return {
        "tool_name": tr.get("tool_name"),
        "available_columns": cols,
        "dimension_columns": dims,
        "default_sort_column": rightmost_col,
        "totals_marker": totals_marker,
        "notes": [
            "If sort column is omitted, backend will use default_sort_column.",
            "include_totals controls whether rows marked with totals_marker are kept.",
        ],
    }


def _apply_partial_spec(
    base: FormatSpec,
    raw: Dict[str, Any],
) -> Tuple[FormatSpec, List[str]]:
    """
    Best-effort merge:
    - Apply only fields that validate
    - Collect notes for invalid/unknown inputs
    """
    notes: List[str] = []
    updates: Dict[str, Any] = {}

    # unit
    if "unit" in raw:
        try:
            # Let pydantic validator normalize aliases by constructing a tiny model
            updates["unit"] = FormatSpec.model_validate({"unit": raw.get("unit")}).unit
        except Exception:
            notes.append(f"Ignored invalid unit: {raw.get('unit')!r}")

    # decimals
    if "decimals" in raw:
        try:
            d = int(raw.get("decimals"))
            if 0 <= d <= 3:
                updates["decimals"] = d
            else:
                notes.append(f"Ignored decimals outside 0..3: {raw.get('decimals')!r}")
        except Exception:
            notes.append(f"Ignored invalid decimals: {raw.get('decimals')!r}")

    # top_n
    if "top_n" in raw:
        try:
            n = raw.get("top_n")
            if n is None:
                updates["top_n"] = None
            else:
                n_i = int(n)
                if 1 <= n_i <= 100:
                    updates["top_n"] = n_i
                else:
                    notes.append(f"Ignored top_n outside 1..100: {raw.get('top_n')!r}")
        except Exception:
            notes.append(f"Ignored invalid top_n: {raw.get('top_n')!r}")

    # include_totals
    if "include_totals" in raw:
        v = raw.get("include_totals")
        if isinstance(v, bool):
            updates["include_totals"] = v
        elif v is None:
            updates["include_totals"] = None
        else:
            notes.append(f"Ignored invalid include_totals: {raw.get('include_totals')!r}")

    # sort
    if "sort" in raw:
        s = raw.get("sort")
        if s is None:
            updates["sort"] = None
        elif isinstance(s, list):
            cleaned: List[Dict[str, Any]] = []
            for item in s[:3]:
                if not isinstance(item, dict):
                    continue
                col = item.get("col")
                direction = (item.get("dir") or item.get("direction") or "desc")
                dir_s = str(direction).strip().lower()
                if dir_s not in {"asc", "desc"}:
                    notes.append(f"Ignored invalid sort dir: {direction!r}")
                    dir_s = "desc"
                col_s = None if col is None else str(col).strip() or None
                order_v = item.get("order")
                order_clean: Optional[List[str]] = None
                if isinstance(order_v, list):
                    tmp: List[str] = []
                    for x in order_v:
                        xs = str(x).strip()
                        if not xs:
                            continue
                        tmp.append(xs)
                        if len(tmp) >= 20:
                            break
                    order_clean = tmp or None
                cleaned.append({"col": col_s, "dir": dir_s, "order": order_clean})
            updates["sort"] = cleaned or None
        else:
            notes.append("Ignored invalid sort (must be list).")

    # rename_columns
    if "rename_columns" in raw:
        rc = raw.get("rename_columns")
        if rc is None:
            updates["rename_columns"] = None
        elif isinstance(rc, dict):
            cleaned: Dict[str, str] = {}
            for k, v in list(rc.items())[:20]:
                k2 = str(k).strip()
                v2 = str(v).strip()
                if not k2 or not v2:
                    continue
                cleaned[k2] = v2
            updates["rename_columns"] = cleaned or None
        else:
            notes.append("Ignored invalid rename_columns (must be object).")

    # derive
    if "derive" in raw:
        dv = raw.get("derive")
        if dv is None:
            updates["derive"] = None
        elif isinstance(dv, list):
            cleaned: List[Dict[str, Any]] = []
            for item in dv[:5]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                op = str(item.get("op") or "").strip()
                a = str(item.get("a") or "").strip()
                b_raw = item.get("b")
                b = None if b_raw is None else str(b_raw or "").strip()
                inputs = item.get("inputs")
                by = item.get("by")
                if not name or not op or not a:
                    # sum may omit a
                    if str(op).strip() != "sum":
                        continue
                # normalize op synonyms
                if op in {"diff", "delta", "subtract"}:
                    op = "sub"
                if op in {"pct", "percent", "percentage"}:
                    op = "pct_change"
                if op in {"divide"}:
                    op = "div"
                if op in {"ratio", "kvot"}:
                    op = "ratio"
                if op not in {"sub", "add", "mul", "div", "ratio", "pct_change", "abs", "neg", "sum"}:
                    notes.append(f"Ignored unknown derive op: {op!r}")
                    continue
                unary = {"abs", "neg"}
                if op in unary:
                    cleaned.append({"name": name, "op": op, "a": a})
                elif op == "sum":
                    if isinstance(inputs, list):
                        ins = [str(x).strip() for x in inputs if str(x).strip()]
                    else:
                        ins = []
                    if len(ins) < 2:
                        continue
                    cleaned.append({"name": name, "op": "sum", "inputs": ins[:10]})
                else:
                    if not b:
                        continue
                    cleaned.append({"name": name, "op": op, "a": a, "b": b})
            updates["derive"] = cleaned or None
        else:
            notes.append("Ignored invalid derive (must be list).")

    # column_decimals
    if "column_decimals" in raw:
        cd = raw.get("column_decimals")
        if cd is None:
            updates["column_decimals"] = None
        elif isinstance(cd, dict):
            cleaned_cd: Dict[str, int] = {}
            for k, v in list(cd.items())[:20]:
                k2 = str(k).strip()
                if not k2:
                    continue
                try:
                    d = int(v)
                except Exception:
                    continue
                if 0 <= d <= 3:
                    cleaned_cd[k2] = d
            updates["column_decimals"] = cleaned_cd or None
        else:
            notes.append("Ignored invalid column_decimals (must be object).")

    # filters
    if "filters" in raw:
        fv = raw.get("filters")
        if fv is None:
            updates["filters"] = None
        elif isinstance(fv, list):
            cleaned_f: List[Dict[str, Any]] = []
            for item in fv[:10]:
                if not isinstance(item, dict):
                    continue
                col = str(item.get("col") or "").strip()
                op = str(item.get("op") or "").strip()
                val = item.get("value")
                fid = item.get("id")
                if not col or not op:
                    continue
                if op not in {"eq", "neq", "in", "not_in", "contains", "gt", "gte", "lt", "lte"}:
                    notes.append(f"Ignored unknown filter op: {op!r}")
                    continue
                cleaned_f.append({"id": (str(fid).strip() if fid is not None else None), "col": col, "op": op, "value": val})
            updates["filters"] = cleaned_f or None
        else:
            notes.append("Ignored invalid filters (must be list).")

    # filter_groups
    if "filter_groups" in raw:
        gv = raw.get("filter_groups")
        if gv is None:
            updates["filter_groups"] = None
        elif isinstance(gv, list):
            cleaned_g: List[Dict[str, Any]] = []
            for g in gv[:5]:
                if not isinstance(g, dict):
                    continue
                op = str(g.get("op") or "or").strip().lower()
                if op not in {"and", "or"}:
                    op = "or"
                rules = g.get("rules") if isinstance(g.get("rules"), list) else []
                rules = [r for r in rules if isinstance(r, dict)]
                rr: List[Dict[str, Any]] = []
                for r in rules[:20]:
                    col = str(r.get("col") or "").strip()
                    rop = str(r.get("op") or "").strip()
                    val = r.get("value")
                    if not col or not rop:
                        continue
                    if rop not in {"eq", "neq", "in", "not_in", "contains", "gt", "gte", "lt", "lte"}:
                        continue
                    rr.append({"col": col, "op": rop, "value": val})
                if not rr:
                    continue
                gid = g.get("id")
                cleaned_g.append({"id": (str(gid).strip() if gid is not None else None), "op": op, "rules": rr})
            updates["filter_groups"] = cleaned_g or None
        else:
            notes.append("Ignored invalid filter_groups (must be list).")

    # filter_expr
    if "filter_expr" in raw:
        fe = raw.get("filter_expr")
        if fe is None:
            updates["filter_expr"] = None
        elif isinstance(fe, dict):
            # minimal shape validation; deeper safety in FormatSpec._count_filter_expr_nodes
            updates["filter_expr"] = fe
        else:
            notes.append("Ignored invalid filter_expr (must be object).")

    # IMPORTANT:
    # - `model_copy(update=...)` does NOT validate/coerce types.
    # - We therefore validate by running FormatSpec.model_validate on the merged dict.
    merged: Dict[str, Any] = base.model_dump(mode="json")
    merged.update(updates)
    try:
        out = FormatSpec.model_validate(merged)
        return out, notes
    except Exception as e:
        notes.append(f"Ignored some invalid fields (validation failed): {type(e).__name__}: {e}")
        return base, notes


def apply_partial_format_overrides(
    *,
    base: FormatSpec,
    raw: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Public helper: apply a possibly-partial dict on top of an existing FormatSpec, best-effort.
    Returns {"spec": FormatSpec, "notes": [..]}.
    """
    if not raw:
        return {"spec": base, "notes": []}
    if not isinstance(raw, dict):
        return {"spec": base, "notes": ["Ignored format_spec override: not an object/dict."]}
    spec2, notes = _apply_partial_spec(base, raw)
    return {"spec": spec2, "notes": notes}


def interpret_format_request_openai(
    *,
    source_tool_run_id: str,
    format_request: str,
    base_spec: Optional[Dict[str, Any]] = None,
    timeout_s: int = 20,
) -> Dict[str, Any]:
    """
    Interpret free-text format_request into a FormatSpec using an OpenAI-compatible API.

    Returns:
      {
        "partial": <dict>,   # only fields that should be applied on top of base_spec
        "reset": <bool>,
        "notes": [ ... ]     # includes LLM notes + validation notes
      }
    """
    fr = (format_request or "").strip()
    if not fr:
        return {"partial": {}, "reset": False, "notes": []}

    ctx = _build_context_from_tool_run(source_tool_run_id=source_tool_run_id)
    rename_map_hint, rename_notes = _extract_rename_columns_from_text(fr)
    derive_hint, derive_notes = _extract_derive_from_text(fr)
    reset_hint = bool(_RE_RESET.search(fr))
    # Deterministic column-decimals hints from text (scalable selectors)
    col_decimals_hint: Dict[str, int] = {}
    m_value = _RE_VALUE_DECIMALS_SV.search(fr)
    if m_value:
        col_decimals_hint["__VALUE__"] = int(m_value.group("d"))
    for m in _RE_COL_DEC_LIST_SV.finditer(fr):
        col_decimals_hint[str(m.group("col")).strip()] = int(m.group("d"))
    m_exact = _RE_COL_DEC_EXACT_SV.search(fr)
    if m_exact:
        col_decimals_hint[str(m_exact.group("col")).strip()] = int(m_exact.group("d"))

    base_url, api_key = _get_openai_cfg()
    url = f"{base_url}/v1/chat/completions"

    system = (
        "You are a formatting interpreter for finance tables.\n"
        "Convert the user's free-text formatting request into a JSON object containing ONLY the fields the user explicitly asked to change.\n"
        "Rules:\n"
        "- Output MUST be valid JSON only (no markdown, no explanations).\n"
        "- Only use these keys: unit, decimals, column_decimals, top_n, sort, include_totals, rename_columns, derive, filters, filter_groups, filter_expr, reset, notes.\n"
        "- IMPORTANT: Do NOT include a key unless the user clearly requested changing it.\n"
        "- If the user asks to reset/clear formatting, set reset=true and omit other keys (or include only what they explicitly re-apply).\n"
        "- column_decimals is an object {column_name: decimals(0..3)} for per-column rounding overrides.\n"
        "  You may also use these scalable selectors as keys:\n"
        "  - __VALUE__ : all value columns (e.g. period columns)\n"
        "  - __PCT__   : all derived percent columns (pct_change) by their derive name\n"
        "  - re:<pattern> : regex applied to column names, e.g. re:^2025-\n"
        "- rename_columns must be an object mapping old column names to new display names.\n"
        "- derive must be a list of objects.\n"
        "  Supported ops: sub, add, mul, div, ratio, pct_change, abs, neg, sum.\n"
        "  - sub: name, op='sub', a, b  => out = a - b\n"
        "  - add: name, op='add', a, b  => out = a + b\n"
        "  - mul: name, op='mul', a, b  => out = a * b\n"
        "  - div/ratio: name, op='div'|'ratio', a, b => out = a / b\n"
        "  - pct_change: name, op='pct_change', a, b => out = (a - b) / b   (fraction; UI renders as percent)\n"
        "  - abs: name, op='abs', a => out = abs(a)\n"
        "  - neg: name, op='neg', a => out = -a\n"
        "  - sum: name, op='sum', inputs:[col1,col2,...] => out = sum(inputs)\n"
        "- If uncertain about a field, omit it and add a note explaining what was unclear.\n"
        "- unit must be one of: sek, tsek, msek (accept aliases like mkr->msek, tkr->tsek, kr->sek).\n"
        "- decimals must be 0..3.\n"
        "- top_n must be 1..100.\n"
        "- sort is a list of objects {col, dir, order?} where dir is asc|desc; col may be null/omitted.\n"
        "  - Optional sort.order (list of strings) means categorical ordering for a string column, e.g.\n"
        "    {\"col\":\"rr_level_1\",\"dir\":\"asc\",\"order\":[\"Intäkter\",\"Kostnader\"]}\n"
        "- filters is a list of objects {col, op, value}.\n"
        "  Supported ops: eq, neq, in, not_in, contains, gt, gte, lt, lte.\n"
        "- filter_groups is a list of objects {op, rules:[{col,op,value}, ...]} where op is and|or.\n"
        "  Top-level semantics: all filter_groups are AND'ed together.\n"
        "- filter_expr is a nested boolean expression tree. Prefer filter_expr for complex logic.\n"
        "  Shapes:\n"
        "    - {\"op\":\"and\"|\"or\",\"args\":[<expr>,...]}\n"
        "    - {\"op\":\"not\",\"arg\":<expr>}\n"
        "    - {\"col\":...,\"op\":...,\"value\":...}  (leaf rule)\n"
        "Example:\n"
        "{\"column_decimals\":{\"pct\":1},\"derive\":[{\"name\":\"pct\",\"op\":\"pct_change\",\"a\":\"2025-02\",\"b\":\"2025-01\"}]}\n"
    )

    user = {
        "format_request": fr,
        "context": ctx,
        "base_spec": base_spec or {},
    }

    payload = {
        "model": _openai_model(),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _safe_json(user)},
        ],
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI-compatible call failed: {resp.status_code} {resp.text}")
    data = resp.json()
    content = (
        (data.get("choices") or [{}])[0]
        .get("message", {})
        .get("content")
    )
    if not isinstance(content, str):
        raise RuntimeError("LLM response missing message.content")

    # Parse JSON-only response
    try:
        raw_obj = json.loads(content)
    except Exception:
        raise RuntimeError(f"LLM did not return valid JSON. content={content[:300]!r}")

    if not isinstance(raw_obj, dict):
        raise RuntimeError("LLM JSON must be an object")

    # Deterministic rename hint: if we can extract a rename mapping, apply it even if LLM misses it.
    if rename_map_hint and "rename_columns" not in raw_obj:
        raw_obj["rename_columns"] = rename_map_hint
    if derive_hint and "derive" not in raw_obj:
        raw_obj["derive"] = derive_hint
    if reset_hint and "reset" not in raw_obj:
        raw_obj["reset"] = True
    if col_decimals_hint and "column_decimals" not in raw_obj:
        raw_obj["column_decimals"] = col_decimals_hint

    # Deterministic filter hints
    filt_hint: List[Dict[str, Any]] = []
    m_eq = _RE_FILTER_EQ_SV.search(fr)
    if m_eq:
        v = (m_eq.group("val") or "").strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1].strip()
        filt_hint.append({"col": m_eq.group("col"), "op": "eq", "value": v})
    m_ex = _RE_FILTER_EXCL_SV.search(fr)
    if m_ex:
        v = (m_ex.group("val") or "").strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1].strip()
        filt_hint.append({"col": m_ex.group("col"), "op": "neq", "value": v})
    m_c = _RE_FILTER_CONTAINS_SV.search(fr)
    if m_c:
        v = (m_c.group("val") or "").strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1].strip()
        filt_hint.append({"col": m_c.group("col"), "op": "contains", "value": v})
    m_n = _RE_FILTER_NUM_SV.search(fr)
    if m_n:
        op_map = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
        filt_hint.append({"col": m_n.group("col"), "op": op_map.get(m_n.group("op"), "gt"), "value": float(m_n.group("num"))})

    # Deterministic OR hint (legacy): build filter_groups
    # - "col=a eller col=b" (same col)
    # - "col1=a eller col2=b" (different cols)
    m_or2 = _RE_FILTER_OR_TWO_EQ_SV.search(fr)
    m_or = _RE_FILTER_OR_EQ_SV.search(fr)
    if (m_or2 or m_or) and not raw_obj.get("filter_groups"):
        def _stripq(s: str) -> str:
            s2 = (s or "").strip()
            if (s2.startswith('"') and s2.endswith('"')) or (s2.startswith("'") and s2.endswith("'")):
                return s2[1:-1].strip()
            return s2
        if m_or2:
            col1 = str(m_or2.group("col1") or "").strip()
            col2 = str(m_or2.group("col2") or "").strip()
            a = _stripq(m_or2.group("a") or "")
            b = _stripq(m_or2.group("b") or "")
            if col1 and col2 and a and b:
                raw_obj["filter_groups"] = [
                    {"op": "or", "rules": [{"col": col1, "op": "eq", "value": a}, {"col": col2, "op": "eq", "value": b}]}
                ]
        else:
            col = str(m_or.group("col") or "").strip()
            a = _stripq(m_or.group("a") or "")
            b = _stripq(m_or.group("b") or "")
            if col and a and b:
                raw_obj["filter_groups"] = [
                    {"op": "or", "rules": [{"col": col, "op": "eq", "value": a}, {"col": col, "op": "eq", "value": b}]}
                ]

    # If we didn't produce an OR-group, fall back to AND-filters
    if not raw_obj.get("filter_groups") and filt_hint and "filters" not in raw_obj:
        raw_obj["filters"] = filt_hint

    # If the user explicitly used OR-language ("eller"/"or") but the model returned flat filters,
    # interpret them as a single OR-group to avoid accidentally AND'ing them.
    if not raw_obj.get("filter_groups"):
        try:
            if ("eller" in fr.lower() or " or " in fr.lower()) and isinstance(raw_obj.get("filters"), list):
                rules = [x for x in raw_obj.get("filters") if isinstance(x, dict)]
                if len(rules) >= 2:
                    raw_obj["filter_groups"] = [{"op": "or", "rules": rules}]
                    raw_obj.pop("filters", None)
        except Exception:
            pass

    # If the user asked for OR (eller/or) and did NOT also ask for AND (och/and),
    # avoid keeping flat filters, since that would AND them and defeat the OR intent.
    try:
        fr_l2 = fr.lower()
        if raw_obj.get("filter_groups") and isinstance(raw_obj.get("filters"), list):
            if ("eller" in fr_l2 or " or " in fr_l2) and ("och" not in fr_l2 and " and " not in fr_l2):
                raw_obj.pop("filters", None)
    except Exception:
        pass

    # If LLM returned filter_expr, we prefer it and drop filters/filter_groups to avoid double-application.
    if isinstance(raw_obj.get("filter_expr"), dict):
        raw_obj.pop("filters", None)
        raw_obj.pop("filter_groups", None)

    llm_notes: List[str] = []
    if isinstance(raw_obj.get("notes"), list):
        llm_notes = [str(x) for x in raw_obj.get("notes") if isinstance(x, (str, int, float)) and str(x).strip()]

    # Decide which keys to keep as "partial updates".
    # We still try to avoid unintended changes, but prefer "reasonable interpretation + notes"
    # over dropping fields due to brittle keyword gating.
    partial: Dict[str, Any] = {}
    fr_l = fr.lower()
    pct_dec_match = _RE_PCT_DECIMALS_SV.search(fr)
    forced_pct_decimals = int(pct_dec_match.group(1)) if pct_dec_match else None
    if "unit" in raw_obj and any(k in fr_l for k in ["mkr", "msek", "tkr", "tsek", "kr", "sek"]):
        partial["unit"] = raw_obj.get("unit")
    if "decimals" in raw_obj and ("decimal" in fr_l or re.search(r"\b\d+\s*decimal", fr_l)):
        partial["decimals"] = raw_obj.get("decimals")
    if "column_decimals" in raw_obj and ("kolumn" in fr_l or "procent" in fr_l or "percent" in fr_l):
        partial["column_decimals"] = raw_obj.get("column_decimals")
    if "top_n" in raw_obj and ("top" in fr_l or "topp" in fr_l):
        partial["top_n"] = raw_obj.get("top_n")
    if "include_totals" in raw_obj and ("total" in fr_l or "summa" in fr_l):
        partial["include_totals"] = raw_obj.get("include_totals")
    if "sort" in raw_obj and ("sort" in fr_l or "sortera" in fr_l or "descending" in fr_l or "asc" in fr_l):
        partial["sort"] = raw_obj.get("sort")
    if "rename_columns" in raw_obj and ("döp" in fr_l or "rename" in fr_l or "kalla" in fr_l):
        partial["rename_columns"] = raw_obj.get("rename_columns")
    if "derive" in raw_obj and (
        "skillnad" in fr_l
        or "diff" in fr_l
        or "delta" in fr_l
        or "kvot" in fr_l
        or "ratio" in fr_l
        or "andel" in fr_l
        or "/" in fr_l
        or "delat" in fr_l
        or "dividerat" in fr_l
        or "abs" in fr_l
        or "absolut" in fr_l
        or "minus" in fr_l
        or "neg" in fr_l
        or "summa" in fr_l
        or "summera" in fr_l
        or "sum(" in fr_l
        or "procent" in fr_l
    ):
        partial["derive"] = raw_obj.get("derive")
    # Filters: be permissive. If the model returns them, we apply them (validated later) even if
    # the user didn't use our exact trigger words.
    if "filters" in raw_obj:
        partial["filters"] = raw_obj.get("filters")
    if "filter_groups" in raw_obj:
        partial["filter_groups"] = raw_obj.get("filter_groups")
    if "filter_expr" in raw_obj:
        partial["filter_expr"] = raw_obj.get("filter_expr")

    # Always honor deterministic hints (even if keywords weren't present due to language variance)
    if rename_map_hint:
        partial["rename_columns"] = raw_obj.get("rename_columns")
    if derive_hint:
        partial["derive"] = raw_obj.get("derive")

    # Deterministic: "procent med X decimal" -> set column_decimals for percent-derived columns
    if forced_pct_decimals is not None:
        pct_cols: List[str] = []
        # from base_spec
        try:
            if isinstance(base_spec, dict):
                for d in (base_spec.get("derive") or []):
                    if isinstance(d, dict) and d.get("op") in {"pct_change"} and d.get("name"):
                        pct_cols.append(str(d.get("name")))
        except Exception:
            pass
        # from incoming derive
        try:
            for d in (partial.get("derive") or []):
                if isinstance(d, dict) and d.get("op") in {"pct_change"} and d.get("name"):
                    pct_cols.append(str(d.get("name")))
        except Exception:
            pass
        pct_cols = [c for c in pct_cols if c]
        if pct_cols:
            cd = partial.get("column_decimals")
            if not isinstance(cd, dict):
                cd = {}
            for c in pct_cols:
                cd[c] = int(forced_pct_decimals)
            partial["column_decimals"] = cd

    # Validate partial keys best-effort (without forcing defaults)
    base_for_validation = merge_with_default(None)
    _, validation_notes = _apply_partial_spec(base_for_validation, partial)

    # Keep LLM notes first, then validation notes
    notes = _detect_unsupported_requests(fr) + rename_notes + derive_notes + llm_notes + validation_notes

    reset_val = bool(raw_obj.get("reset")) or reset_hint
    if reset_val:
        # On reset, ignore other partial keys unless user explicitly re-applied them.
        # (We still keep partial if present; backend may apply them after resetting.)
        pass

    if not partial and not notes and not reset_val:
        notes.append("Jag kunde inte tolka någon formateringsinstruktion från din text.")

    return {"partial": partial, "reset": reset_val, "notes": notes}


