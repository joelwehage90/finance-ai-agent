from __future__ import annotations

import json
import time
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..supabase_client import supabase
from .format_spec import FormatSpec, resolve_missing_sort_columns
from .format_summary import build_format_summary_sv


_DISPLAY_UNIT = {"sek": "kr", "tsek": "tkr", "msek": "mkr"}
_UNIT_DIVISOR = {"sek": 1.0, "tsek": 1_000.0, "msek": 1_000_000.0}


def _res_data(res: Any) -> List[Dict[str, Any]]:
    data = getattr(res, "data", None)
    if data is None:
        data = res.data
    return data or []


def _res_error(res: Any) -> Any:
    return getattr(res, "error", None)


def _estimate_bytes(payload: Any) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _casefold(s: Any) -> str:
    return str(s or "").casefold()


def _resolve_col(col: Any, *, col_map: Dict[str, str]) -> Optional[str]:
    """
    Resolve a user-provided column name to the canonical column key in the current table.
    Case-insensitive.
    """
    if col is None:
        return None
    s = str(col).strip()
    if not s:
        return None
    return col_map.get(s.casefold(), s)


def _infer_columns(rows: List[Dict[str, Any]], explicit_cols: Optional[Sequence[str]]) -> List[str]:
    if explicit_cols:
        return [str(c) for c in explicit_cols]
    if not rows:
        return []
    # Preserve insertion order from the first row (Python 3.7+ dict order)
    return [str(k) for k in rows[0].keys()]


def _split_dims_and_values(meta: Dict[str, Any], columns: Sequence[str], rows: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    dims: List[str] = []
    m_rows = meta.get("rows")
    if isinstance(m_rows, list) and all(isinstance(x, str) for x in m_rows):
        dims = list(m_rows)

    if not dims:
        # Fallback heuristic: treat non-numeric columns in first row as dims
        if rows:
            r0 = rows[0]
            for c in columns:
                v = r0.get(c)
                if _safe_float(v) is None:
                    dims.append(c)

    dim_set = set(dims)
    value_cols = [c for c in columns if c not in dim_set]
    return dims, value_cols


def _drop_total_rows(rows: List[Dict[str, Any]], dims: Sequence[str]) -> List[Dict[str, Any]]:
    # Income statement tool marks totals as "__TOTAL__" in dimension columns.
    out: List[Dict[str, Any]] = []
    for r in rows:
        is_total = False
        for d in dims:
            if str(r.get(d) or "") == "__TOTAL__":
                is_total = True
                break
        if not is_total:
            out.append(r)
    return out


def _find_total_row(rows: List[Dict[str, Any]], dims: Sequence[str]) -> Optional[Dict[str, Any]]:
    """
    Find the first totals row. Income statement marks totals as "__TOTAL__" in any dim column.
    """
    for r in rows:
        for d in dims:
            if str(r.get(d) or "") == "__TOTAL__":
                return r
    return None


def _is_total_row(
    r: Dict[str, Any],
    *,
    dims: Sequence[str],
    totals_marker: str,
    totals_columns: Optional[Sequence[str]] = None,
) -> bool:
    # Tool-neutral explicit flags (preferred when available)
    if r.get("_is_total") is True:
        return True
    if str(r.get("__row_type") or "").lower() == "total":
        return True

    cols = list(totals_columns) if totals_columns else list(dims)
    for d in cols:
        if str(r.get(d) or "") == totals_marker:
            return True
    return False


def _convert_row_unit(row: Dict[str, Any], value_cols: Sequence[str], *, unit: str) -> Dict[str, Any]:
    div = float(_UNIT_DIVISOR.get(unit, 1.0))
    r2 = dict(row)
    for c in value_cols:
        v = _safe_float(r2.get(c))
        if v is None:
            continue
        r2[c] = v / div
    return r2


def _apply_unit_conversion(
    rows: List[Dict[str, Any]],
    value_cols: Sequence[str],
    *,
    unit: str,
) -> List[Dict[str, Any]]:
    div = float(_UNIT_DIVISOR.get(unit, 1.0))
    out: List[Dict[str, Any]] = []
    for r in rows:
        r2 = dict(r)
        for c in value_cols:
            v = _safe_float(r2.get(c))
            if v is None:
                continue
            r2[c] = v / div
        out.append(r2)
    return out


def _apply_rounding(rows: List[Dict[str, Any]], cols: Sequence[str], *, decimals: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        r2 = dict(r)
        for c in cols:
            v = _safe_float(r2.get(c))
            if v is None:
                continue
            r2[c] = round(v, int(decimals))
        out.append(r2)
    return out


def _apply_rounding_by_column(
    rows: List[Dict[str, Any]],
    cols: Sequence[str],
    *,
    default_decimals: int,
    per_column_decimals: Optional[Dict[str, int]],
) -> List[Dict[str, Any]]:
    """
    Round each column in `cols` with an optional per-column override.
    Keeps logic simple while allowing e.g. percent columns to have 1 decimal
    while money columns have 0 decimals.
    """
    if not rows or not cols:
        return rows
    if not per_column_decimals:
        return _apply_rounding(rows, cols, decimals=default_decimals)

    # group columns by decimals to reduce passes
    groups: Dict[int, List[str]] = {}
    for c in cols:
        d = per_column_decimals.get(str(c), default_decimals)
        try:
            d_i = int(d)
        except Exception:
            d_i = default_decimals
        if d_i < 0:
            d_i = 0
        # Allow a slightly higher cap internally (e.g. percent columns stored as fractions may need +2 decimals).
        if d_i > 6:
            d_i = 6
        groups.setdefault(d_i, []).append(str(c))

    out = rows
    for d_i, gcols in groups.items():
        out = _apply_rounding(out, gcols, decimals=d_i)
    return out


def _compute_derived_columns(
    rows: List[Dict[str, Any]],
    *,
    derive: List[Dict[str, Any]],
    notes: List[str],
    dims: Sequence[str] = (),
    total_row: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute derived columns from existing numeric columns.
    Supported ops:
    - sub: a - b
    - add: a + b
    - mul: a * b
    - div/ratio: a / b (safe div0 -> None)
    - pct_change: (a - b) / b  (fraction; safe div0 -> None)
    - abs: abs(a)
    - neg: -a
    - sum: sum(inputs)
    """
    if not rows or not derive:
        return rows

    out: List[Dict[str, Any]] = []
    for r in rows:
        r2 = dict(r)
        for d in derive:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "").strip()
            op = str(d.get("op") or "").strip()
            a = None if d.get("a") is None else str(d.get("a") or "").strip()
            b = None if d.get("b") is None else str(d.get("b") or "").strip()
            inputs = d.get("inputs")
            unary = {"abs", "neg"}
            if not name or not op:
                continue

            # detect if this row is the totals row
            is_total_row = False
            for dim in dims:
                if str(r2.get(dim) or "") == "__TOTAL__":
                    is_total_row = True
                    break

            if op == "sum":
                if not isinstance(inputs, list) or len(inputs) < 2:
                    r2[name] = None
                    continue
                vals: List[float] = []
                ok = True
                for col in inputs[:10]:
                    col_s = str(col).strip()
                    if not col_s:
                        ok = False
                        break
                    v = _safe_float(r2.get(col_s))
                    if v is None:
                        ok = False
                        break
                    vals.append(v)
                r2[name] = sum(vals) if ok else None
            else:
                # binary/unary ops
                if not a:
                    r2[name] = None
                    continue
                av = _safe_float(r2.get(a))
                bv = _safe_float(r2.get(b)) if b else None
                if av is None or (op not in unary and op not in {"abs", "neg"} and bv is None):
                    r2[name] = None
                    continue

                if op == "sub":
                    r2[name] = av - bv
                elif op == "add":
                    r2[name] = av + bv
                elif op == "mul":
                    r2[name] = av * bv
                elif op in {"div", "ratio"}:
                    if bv == 0:
                        r2[name] = None
                    else:
                        r2[name] = av / bv
                elif op == "pct_change":
                    if bv == 0:
                        r2[name] = None
                    else:
                        r2[name] = (av - bv) / bv
                elif op == "abs":
                    r2[name] = abs(av)
                elif op == "neg":
                    r2[name] = -av
                else:
                    # Unknown op: note once (global)
                    msg = f"Ignored unknown derive op: {op}"
                    if msg not in notes:
                        notes.append(msg)
        out.append(r2)
    return out


def _apply_filters(
    rows: List[Dict[str, Any]],
    *,
    filters: List[Dict[str, Any]],
    notes: List[str],
) -> List[Dict[str, Any]]:
    if not rows or not filters:
        return rows

    # Case-insensitive column resolution
    col_map = {str(k).casefold(): str(k) for k in rows[0].keys()}

    def _coerce_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    def _match(r: Dict[str, Any], f: Dict[str, Any]) -> bool:
        col0 = str(f.get("col") or "").strip()
        col = _resolve_col(col0, col_map=col_map)
        op = str(f.get("op") or "").strip()
        val = f.get("value")
        if not col or not op:
            return True
        if col.casefold() not in col_map:
            msg = f"Ignored filter on unknown column: {col0}"
            if msg not in notes:
                notes.append(msg)
            return True

        cell = r.get(col)
        # string-ish comparisons
        if op == "contains":
            if val is None:
                return True
            return str(val).lower() in str(cell or "").lower()
        if op == "eq":
            if val is None:
                return cell is None or str(cell) == ""
            return _casefold(cell) == _casefold(val)
        if op == "neq":
            if val is None:
                return not (cell is None or str(cell) == "")
            return _casefold(cell) != _casefold(val)
        if op == "in":
            vals = [_casefold(x) for x in _coerce_list(val)]
            return _casefold(cell) in set(vals)
        if op == "not_in":
            vals = [_casefold(x) for x in _coerce_list(val)]
            return _casefold(cell) not in set(vals)

        # numeric comparisons
        cell_f = _safe_float(cell)
        val_f = _safe_float(val)
        if cell_f is None or val_f is None:
            return False
        if op == "gt":
            return cell_f > val_f
        if op == "gte":
            return cell_f >= val_f
        if op == "lt":
            return cell_f < val_f
        if op == "lte":
            return cell_f <= val_f

        msg = f"Ignored unknown filter op: {op}"
        if msg not in notes:
            notes.append(msg)
        return True

    before = len(rows)
    out = []
    for r in rows:
        ok = True
        for f in filters:
            if isinstance(f, dict) and not _match(r, f):
                ok = False
                break
        if ok:
            out.append(r)
    dropped = before - len(out)
    if dropped > 0:
        notes.append(f"Applied filters: dropped {dropped} rows.")
    return out


def _apply_filter_groups(
    rows: List[Dict[str, Any]],
    *,
    groups: List[Dict[str, Any]],
    notes: List[str],
) -> List[Dict[str, Any]]:
    """
    Apply filter groups with OR/AND inside each group.
    Top-level semantics: AND across groups (row must satisfy every group).
    """
    if not rows or not groups:
        return rows

    col_map = {str(k).casefold(): str(k) for k in rows[0].keys()}

    def _coerce_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    def _match_rule(r: Dict[str, Any], f: Dict[str, Any]) -> bool:
        col0 = str(f.get("col") or "").strip()
        col = _resolve_col(col0, col_map=col_map)
        op = str(f.get("op") or "").strip()
        val = f.get("value")
        if not col or not op:
            return True
        if col.casefold() not in col_map:
            msg = f"Ignored filter on unknown column: {col0}"
            if msg not in notes:
                notes.append(msg)
            return True

        cell = r.get(col)
        if op == "contains":
            if val is None:
                return True
            return str(val).lower() in str(cell or "").lower()
        if op == "eq":
            if val is None:
                return cell is None or str(cell) == ""
            return _casefold(cell) == _casefold(val)
        if op == "neq":
            if val is None:
                return not (cell is None or str(cell) == "")
            return _casefold(cell) != _casefold(val)
        if op == "in":
            vals = [_casefold(x) for x in _coerce_list(val)]
            return _casefold(cell) in set(vals)
        if op == "not_in":
            vals = [_casefold(x) for x in _coerce_list(val)]
            return _casefold(cell) not in set(vals)

        cell_f = _safe_float(cell)
        val_f = _safe_float(val)
        if cell_f is None or val_f is None:
            return False
        if op == "gt":
            return cell_f > val_f
        if op == "gte":
            return cell_f >= val_f
        if op == "lt":
            return cell_f < val_f
        if op == "lte":
            return cell_f <= val_f

        msg = f"Ignored unknown filter op: {op}"
        if msg not in notes:
            notes.append(msg)
        return True

    def _eval_group(r: Dict[str, Any], g: Dict[str, Any]) -> bool:
        op = str(g.get("op") or "or").strip().lower()
        rules = g.get("rules") if isinstance(g.get("rules"), list) else []
        rules = [x for x in rules if isinstance(x, dict)]
        if not rules:
            return True
        if op == "and":
            return all(_match_rule(r, f) for f in rules)
        # default OR
        return any(_match_rule(r, f) for f in rules)

    before = len(rows)
    out: List[Dict[str, Any]] = []
    for r in rows:
        ok = True
        for g in groups:
            if isinstance(g, dict) and not _eval_group(r, g):
                ok = False
                break
        if ok:
            out.append(r)
    dropped = before - len(out)
    if dropped > 0:
        notes.append(f"Applied filter groups: dropped {dropped} rows.")
    return out


def _apply_filter_expr(
    rows: List[Dict[str, Any]],
    *,
    expr: Dict[str, Any],
    notes: List[str],
) -> List[Dict[str, Any]]:
    """
    Apply a nested boolean filter expression tree.

    Supported shapes:
    - {"op":"and"|"or", "args":[expr,...]}
    - {"op":"not", "arg":expr}
    - {"col":..., "op":..., "value":...}   # leaf rule
    - {"rule": {...}}                      # leaf wrapper
    """
    if not rows or not expr or not isinstance(expr, dict):
        return rows

    col_map = {str(k).casefold(): str(k) for k in rows[0].keys()}

    def _coerce_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    def _match_leaf(r: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        col0 = str(rule.get("col") or "").strip()
        col = _resolve_col(col0, col_map=col_map)
        op = str(rule.get("op") or "").strip()
        val = rule.get("value")
        if not col or not op:
            return True
        if col.casefold() not in col_map:
            msg = f"Ignored filter on unknown column: {col0}"
            if msg not in notes:
                notes.append(msg)
            return True

        cell = r.get(col)
        if op == "contains":
            if val is None:
                return True
            return str(val).lower() in str(cell or "").lower()
        if op == "eq":
            if val is None:
                return cell is None or str(cell) == ""
            return _casefold(cell) == _casefold(val)
        if op == "neq":
            if val is None:
                return not (cell is None or str(cell) == "")
            return _casefold(cell) != _casefold(val)
        if op == "in":
            vals = [_casefold(x) for x in _coerce_list(val)]
            return _casefold(cell) in set(vals)
        if op == "not_in":
            vals = [_casefold(x) for x in _coerce_list(val)]
            return _casefold(cell) not in set(vals)

        cell_f = _safe_float(cell)
        val_f = _safe_float(val)
        if cell_f is None or val_f is None:
            return False
        if op == "gt":
            return cell_f > val_f
        if op == "gte":
            return cell_f >= val_f
        if op == "lt":
            return cell_f < val_f
        if op == "lte":
            return cell_f <= val_f

        msg = f"Ignored unknown filter op: {op}"
        if msg not in notes:
            notes.append(msg)
        return True

    def _eval(r: Dict[str, Any], e: Any) -> bool:
        if e is None:
            return True
        if isinstance(e, dict) and "rule" in e and isinstance(e.get("rule"), dict):
            return _eval(r, e.get("rule"))
        if not isinstance(e, dict):
            return True
        op = str(e.get("op") or "").lower()
        if op in {"and", "or"}:
            args = e.get("args") if isinstance(e.get("args"), list) else []
            args = list(args) if args else []
            if not args:
                return True
            if op == "and":
                return all(_eval(r, a) for a in args)
            return any(_eval(r, a) for a in args)
        if op == "not":
            return not _eval(r, e.get("arg"))
        # leaf
        if "col" in e and "op" in e:
            return _match_leaf(r, e)
        return True

    before = len(rows)
    out = [r for r in rows if _eval(r, expr)]
    dropped = before - len(out)
    if dropped > 0:
        notes.append(f"Applied filter_expr: dropped {dropped} rows.")
    return out


def _sort_rows(rows: List[Dict[str, Any]], sort_keys: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic, stable multi-key sort.

    Supports two kinds of sort keys:
    - Numeric/string column sort: {"col": "...", "dir": "asc"|"desc"}
    - Categorical list ordering: {"col": "...", "dir": "asc"|"desc", "order": ["Intäkter","Kostnader",...]}
      Matching is case-insensitive substring match (token in cell_value).
      Non-matching values are placed after the list and KEEP ORIGINAL ORDER (stable).
    """
    out = list(rows)
    if not out or not sort_keys:
        return out

    # Base order snapshot for stable "unmatched keep order" semantics.
    base_index: Dict[int, int] = {id(r): i for i, r in enumerate(out)}

    # Apply stable sorts from last key to first key (like SQL ORDER BY).
    for sk in reversed(list(sort_keys)):
        col = sk.get("col")
        if not col:
            continue
        direction = str(sk.get("dir") or "desc").lower()
        reverse = direction == "desc"

        order_v = sk.get("order")
        if isinstance(order_v, list) and order_v:
            # Categorical list ordering (primary/secondary supported via stable sort).
            tokens = [str(x).strip() for x in order_v if str(x).strip()]
            if not tokens:
                continue
            tokens_cf = [t.casefold() for t in tokens]
            if reverse:
                tokens_cf = list(reversed(tokens_cf))

            def key_fn_cat(r: Dict[str, Any]):
                v = r.get(col)
                s = str(v or "").casefold()
                match_idx: Optional[int] = None
                for i2, tok in enumerate(tokens_cf):
                    if tok and tok in s:
                        match_idx = i2
                        break
                if match_idx is not None:
                    # Matched: allow previous sorts to decide within group (stable).
                    return (0, match_idx, 0)
                # Unmatched: place after list and preserve original row order.
                return (1, base_index.get(id(r), 10**9), 0)

            out.sort(key=key_fn_cat, reverse=False)
            continue

        # Default: sort by numeric when possible, otherwise by string (casefold).
        def key_fn_default(r: Dict[str, Any]):
            v = r.get(col)
            is_none = v is None or (isinstance(v, str) and not v.strip())
            # Make None last for BOTH asc/desc by flipping the None-rank depending on reverse.
            none_rank = (-1 if reverse else 1) if is_none else 0
            if is_none:
                return (none_rank, 0, 0)
            vf = _safe_float(v)
            if vf is not None:
                return (0, 0, vf)
            return (0, 1, str(v).casefold())

        out.sort(key=key_fn_default, reverse=reverse)

    return out


def _expand_column_decimals(
    *,
    raw: Optional[Dict[str, Any]],
    columns: Sequence[str],
    value_cols: Sequence[str],
    derive_specs: Sequence[Dict[str, Any]],
    notes: List[str],
) -> Optional[Dict[str, int]]:
    """
    Expand a compact/scalable column_decimals mapping into explicit per-column overrides.

    Supported keys:
    - Exact column names: {"2025-02": 2}
    - "__VALUE__": applies to all value columns (non-dim columns)
    - "__PCT__": applies to derived percent columns whose op is pct_change (by derive name)
    - "re:<pattern>": regex applied to column names, e.g. {"re:^2025-": 0}

    Unknown keys are ignored with a note.
    """
    if not raw or not isinstance(raw, dict):
        return None

    # case-insensitive column resolution (includes derived names)
    col_map: Dict[str, str] = {str(c).casefold(): str(c) for c in columns}
    # Also allow targeting derived columns by name (they don't exist in the raw tool-run columns list).
    for ds in derive_specs:
        if not isinstance(ds, dict):
            continue
        name = ds.get("name")
        if name is None:
            continue
        n = str(name).strip()
        if n:
            col_map[n.casefold()] = n
    out: Dict[str, int] = {}

    def _coerce_dec(v: Any) -> Optional[int]:
        try:
            d = int(v)
        except Exception:
            return None
        if d < 0:
            d = 0
        if d > 3:
            d = 3
        return d

    # Apply in insertion order (Python dict order) so later keys win.
    for k, v in raw.items():
        key = str(k).strip()
        d = _coerce_dec(v)
        if not key or d is None:
            continue

        if key == "__VALUE__":
            for c in value_cols:
                out[str(c)] = d
            continue

        if key == "__PCT__":
            for ds in derive_specs:
                if not isinstance(ds, dict):
                    continue
                if ds.get("op") in {"pct_change"} and ds.get("name"):
                    out[str(ds["name"])] = d
            continue

        if key.lower().startswith("re:"):
            pat = key[3:].strip()
            if not pat:
                continue
            try:
                rx = re.compile(pat)
            except Exception:
                notes.append(f"Ignored invalid column_decimals regex: {pat!r}")
                continue
            matched = 0
            for c in columns:
                cs = str(c)
                if rx.search(cs):
                    out[cs] = d
                    matched += 1
            if matched == 0:
                notes.append(f"column_decimals regex matched no columns: {pat!r}")
            continue

        # exact column
        resolved = col_map.get(key.casefold())
        if resolved:
            out[resolved] = d
            continue

        notes.append(f"Ignored column_decimals for unknown column/selector: {key!r}")

    return out or None


def _build_single_presentation_table_payload(
    *,
    meta: Dict[str, Any],
    table_rows: List[Dict[str, Any]],
    explicit_cols: Optional[List[str]],
    spec: FormatSpec,
    include_summary: bool,
    table_key: Optional[str] = None,
) -> tuple[Dict[str, Any], FormatSpec, int]:
    """
    Build the standard single-table payload, given already-materialized rows.
    Returns: (payload, applied_spec, row_count)
    """
    table_rows = [r for r in (table_rows or []) if isinstance(r, dict)]

    # Prefer explicit column ordering from meta (dims + value_columns) when available.
    # This keeps variance tables stable and also works for empty tables.
    meta_rows = meta.get("rows") if isinstance(meta.get("rows"), list) else None
    meta_rows = [str(x) for x in meta_rows if x is not None] if meta_rows else None
    meta_vals = meta.get("value_columns") if isinstance(meta.get("value_columns"), list) else None
    meta_vals = [str(x) for x in meta_vals if x is not None] if meta_vals else None

    columns = _infer_columns(table_rows, explicit_cols)
    if meta_rows and meta_vals:
        # If we have no row sample, synthesize columns entirely.
        if not columns:
            columns = list(meta_rows) + list(meta_vals)
        else:
            # Reorder columns but keep any unknown extras at the end.
            desired = [c for c in (list(meta_rows) + list(meta_vals)) if c in columns]
            extras = [c for c in columns if c not in set(desired)]
            columns = desired + extras

    dims, value_cols = _split_dims_and_values(meta, columns, table_rows)

    notes: List[str] = []

    # totals marker config (tool-neutral; tools may set meta.totals_marker/label/columns)
    totals_marker = str(meta.get("totals_marker") or "__TOTAL__")
    totals_label = str(meta.get("totals_label") or "Totalt")
    totals_columns = meta.get("totals_columns") if isinstance(meta.get("totals_columns"), list) else None
    totals_columns = [str(x) for x in totals_columns if x is not None] if totals_columns else None

    # include_totals: drop rows marked as totals
    raw_rows_for_totals = list(table_rows)
    total_row_raw = _find_total_row(raw_rows_for_totals, dims) if dims else None
    if spec.include_totals is False and dims:
        before = len(table_rows)
        table_rows = _drop_total_rows(table_rows, dims)
        dropped = before - len(table_rows)
        if dropped:
            notes.append(f"Dropped {dropped} totals rows (include_totals=false).")

    # Apply unit conversion on value columns only (NO rounding yet).
    # IMPORTANT: percent/fraction columns (e.g. explain_pct) must NOT be unit-scaled.
    percent_cols: set[str] = set()
    try:
        if isinstance(meta.get("column_formats"), dict):
            # Resolve percent columns case-insensitively against the inferred `columns` list.
            col_map2 = {str(c).casefold(): str(c) for c in columns}
            for k, v in (meta.get("column_formats") or {}).items():
                if not isinstance(v, dict):
                    continue
                if str(v.get("kind") or "").strip().lower() != "percent":
                    continue
                k2 = str(k).strip()
                if not k2:
                    continue
                percent_cols.add(col_map2.get(k2.casefold(), k2))
    except Exception:
        percent_cols = set()

    unit_value_cols = [c for c in value_cols if c not in percent_cols]
    table_rows = _apply_unit_conversion(table_rows, unit_value_cols, unit=spec.unit)
    total_row_conv = _convert_row_unit(total_row_raw, unit_value_cols, unit=spec.unit) if isinstance(total_row_raw, dict) else None

    # Tool-provided default sort (only if the user didn't specify a concrete sort column).
    # NOTE: default_format_spec() uses sort=[{"col": None, "dir": "desc"}] as a placeholder,
    # which should still allow tools (e.g. variance_tool) to provide a domain-specific default.
    spec_for_sort = spec
    sort_has_concrete_col = False
    if spec.sort:
        for sk in spec.sort:
            try:
                col = sk.col if hasattr(sk, "col") else None
            except Exception:
                col = None
            if col is not None:
                sort_has_concrete_col = True
                break

    if (not spec.sort) or (not sort_has_concrete_col):
        default_sort = meta.get("default_sort")
        if isinstance(default_sort, list):
            try:
                tmp = spec.model_dump(mode="json")
                tmp["sort"] = default_sort
                spec_for_sort = FormatSpec.model_validate(tmp)
            except Exception:
                pass

    # Resolve missing sort columns and sort (if any)
    spec2 = resolve_missing_sort_columns(spec_for_sort, columns=columns)
    sort_dump = [s.model_dump() for s in (spec2.sort or [])]
    # Ignore sort keys that reference unknown columns (v1). We'll add a note.
    col_map = {str(c).casefold(): str(c) for c in columns}
    known_cols = set(columns)
    filtered_sort: List[Dict[str, Any]] = []
    for sk in sort_dump:
        c = sk.get("col")
        if c is None:
            continue
        # case-insensitive resolve
        if isinstance(c, str):
            c_res = col_map.get(c.casefold())
            if c_res:
                c = c_res
                sk["col"] = c_res
        if c not in known_cols:
            notes.append(f"Ignored unknown sort column: {c}")
            continue
        filtered_sort.append(sk)

    # Derive (computed AFTER unit conversion but BEFORE rounding)
    derive_cols: List[str] = []
    if spec2.derive:
        derive_dump = [d.model_dump(mode="json") for d in (spec2.derive or [])]
        # Validate references (against current row keys), case-insensitive
        if table_rows:
            row_keys_map = {str(k).casefold(): str(k) for k in table_rows[0].keys()}
        else:
            row_keys_map = {str(k).casefold(): str(k) for k in columns}
        for d in derive_dump:
            name = str(d.get("name") or "").strip()
            a_raw = d.get("a")
            a = None if a_raw is None else str(a_raw or "").strip()
            op = str(d.get("op") or "").strip()
            b_raw = d.get("b")
            b = None if b_raw is None else str(b_raw or "").strip()
            inputs = d.get("inputs")
            if not name:
                continue
            if op == "sum":
                if not isinstance(inputs, list) or len(inputs) < 2:
                    notes.append(f"Ignored derive '{name}': sum requires inputs (>=2)")
                    continue
                resolved_inputs: List[str] = []
                bad: List[str] = []
                for x in inputs:
                    xs = str(x).strip()
                    if not xs:
                        continue
                    xr = row_keys_map.get(xs.casefold())
                    if not xr:
                        bad.append(xs)
                    else:
                        resolved_inputs.append(xr)
                if bad:
                    notes.append(f"Ignored derive '{name}': unknown sum inputs: {', '.join(bad[:5])}")
                    continue
                d["inputs"] = resolved_inputs
            else:
                if a:
                    ar = row_keys_map.get(a.casefold())
                    if not ar:
                        notes.append(f"Ignored derive '{name}': unknown column a='{a}'")
                        continue
                    d["a"] = ar
                    a = ar
                if op not in {"abs", "neg"}:
                    if not b:
                        notes.append(f"Ignored derive '{name}': missing operand b for op='{op}'")
                        continue
                    br = row_keys_map.get(b.casefold())
                    if not br:
                        notes.append(f"Ignored derive '{name}': unknown column b='{b}'")
                        continue
                    d["b"] = br
            derive_cols.append(name)
        if derive_cols:
            table_rows = _compute_derived_columns(table_rows, derive=derive_dump, notes=notes, dims=dims, total_row=total_row_conv)

    # Filters (after derive, before sort/top_n)
    if getattr(spec2, "filter_expr", None) and isinstance(spec2.filter_expr, dict):
        table_rows = _apply_filter_expr(table_rows, expr=spec2.filter_expr, notes=notes)
    else:
        filters_dump = [f.model_dump(mode="json") for f in (spec2.filters or [])] if spec2.filters else []
        groups_dump = [g.model_dump(mode="json") for g in (spec2.filter_groups or [])] if getattr(spec2, "filter_groups", None) else []
        if filters_dump and groups_dump:
            groups_dump = [{"op": "and", "rules": filters_dump}] + groups_dump
            notes.append("Merged legacy filters into filter_groups (AND-group).")
            filters_dump = []
        if filters_dump:
            table_rows = _apply_filters(table_rows, filters=filters_dump, notes=notes)
        if groups_dump:
            table_rows = _apply_filter_groups(table_rows, groups=groups_dump, notes=notes)

    tool_name = str(meta.get("_tool_name") or meta.get("tool_name") or "").strip()
    apply_income_default = tool_name == "income_statement_tool" and not filtered_sort
    default_sort_note: Optional[str] = None
    if apply_income_default:
        # Keep totals row(s) last, even after sorting.
        cols_for_totals = totals_columns or list(dims)
        non_total_rows: List[Dict[str, Any]] = []
        total_rows: List[Dict[str, Any]] = []
        for r in table_rows:
            if isinstance(r, dict) and _is_total_row(
                r,
                dims=dims,
                totals_marker=totals_marker,
                totals_columns=cols_for_totals,
            ):
                total_rows.append(r)
            else:
                non_total_rows.append(r)

        # 1) Sort by absolute value of the rightmost value column (desc).
        if value_cols:
            sort_col = value_cols[-1]
            default_sort_note = f"konto_typ asc(intäkter,kostnader), abs({sort_col}) desc"

            def _abs_key(row: Dict[str, Any]):
                v = _safe_float(row.get(sort_col))
                if v is None:
                    return (1, 0.0)  # None last
                return (0, -abs(v))

            non_total_rows = sorted(non_total_rows, key=_abs_key)

        # 2) Group by konto_typ in a stable, categorical order.
        if "konto_typ" in columns:
            non_total_rows = _sort_rows(
                non_total_rows,
                [
                    {
                        "col": "konto_typ",
                        "dir": "asc",
                        "order": ["intäkter", "kostnader"],
                    }
                ],
            )

        table_rows = non_total_rows + total_rows

    if filtered_sort:
        table_rows = _sort_rows(table_rows, filtered_sort)

    # top_n
    if spec2.top_n:
        n = int(spec2.top_n)
        if len(table_rows) > n:
            table_rows = table_rows[:n]
            notes.append(f"Applied top_n={n}.")

    # bounds
    if len(table_rows) > 100:
        notes.append(f"Source had {len(table_rows)} rows; showing first 100 rows.")
        table_rows = table_rows[:100]

    # Keep columns bounded too (v1)
    out_columns = list(dims) + list(value_cols) + [c for c in derive_cols if c not in dims and c not in value_cols]
    if len(out_columns) > 12:
        notes.append(f"Source had {len(out_columns)} columns; showing first 12 columns.")
        out_columns = out_columns[:12]
        keep = set(out_columns)
        table_rows = [{k: v for k, v in r.items() if k in keep} for r in table_rows]

    # Final rounding step (presentation-only)
    derive_dump_for_rounding = [d.model_dump(mode="json") for d in (spec2.derive or [])]
    per_col_raw = spec2.column_decimals if isinstance(spec2.column_decimals, dict) else None
    per_col = _expand_column_decimals(raw=per_col_raw, columns=columns, value_cols=value_cols, derive_specs=derive_dump_for_rounding, notes=notes)
    table_rows = _apply_rounding_by_column(table_rows, list(value_cols), default_decimals=spec2.decimals, per_column_decimals=per_col)
    if derive_cols:
        pct_cols = [str(d.get("name")) for d in derive_dump_for_rounding if isinstance(d, dict) and d.get("op") == "pct_change" and d.get("name")]
        per_col_derived: Optional[Dict[str, int]] = dict(per_col) if isinstance(per_col, dict) else None
        if pct_cols:
            if per_col_derived is None:
                per_col_derived = {}
            for c in pct_cols:
                base_pct_dec = per_col.get(c, spec2.decimals) if isinstance(per_col, dict) else spec2.decimals
                try:
                    base_pct_dec_i = int(base_pct_dec)
                except Exception:
                    base_pct_dec_i = int(spec2.decimals)
                per_col_derived[c] = base_pct_dec_i + 2
        table_rows = _apply_rounding_by_column(table_rows, derive_cols, default_decimals=spec2.decimals, per_column_decimals=per_col_derived)

    payload: Dict[str, Any] = {
        "kind": "table",
        "columns": out_columns,
        "rows": table_rows,
        "format": {
            "unit": _DISPLAY_UNIT.get(spec2.unit, spec2.unit),
            "unit_canonical": spec2.unit,
            "decimals": spec2.decimals,
            "column_decimals": per_col,
            "column_formats": None,
            "sorted_by": (", ".join([f"{x['col']} {x.get('dir','desc')}" for x in filtered_sort]) if filtered_sort else None),
            "default_sorted_by": default_sort_note,
            "row_limit": int(spec2.top_n) if spec2.top_n else None,
            "include_totals": spec2.include_totals,
            "totals": {"marker": totals_marker, "label": totals_label, "columns": totals_columns or list(dims)},
        },
    }

    # Column formats: meta.column_formats (preferred) + pct_change derived columns
    try:
        col_formats: Dict[str, Any] = {}
        if isinstance(meta.get("column_formats"), dict):
            for k, v in (meta.get("column_formats") or {}).items():
                if isinstance(v, dict) and str(k).strip():
                    col_formats[str(k)] = v
        if spec2.derive:
            for d in derive_dump_for_rounding:
                if not isinstance(d, dict):
                    continue
                if d.get("op") != "pct_change":
                    continue
                name = str(d.get("name") or "").strip()
                if not name:
                    continue
                dec = per_col.get(name) if isinstance(per_col, dict) else None
                if dec is None:
                    dec = int(spec2.decimals)
                col_formats[name] = {"kind": "percent", "scale": "fraction", "decimals": int(dec)}
        if col_formats:
            payload.setdefault("format", {})
            payload["format"]["column_formats"] = col_formats
    except Exception:
        pass

    if notes:
        payload["notes"] = notes

    # Rename columns (presentation-only)
    if spec2.rename_columns and isinstance(spec2.rename_columns, dict):
        rename_map = {str(k): str(v) for k, v in spec2.rename_columns.items() if str(k).strip() and str(v).strip()}
        if rename_map:
            col_set = set(out_columns)
            col_map2 = {str(c).casefold(): str(c) for c in out_columns}
            resolved_rename: Dict[str, str] = {}
            for k, v in rename_map.items():
                kr = col_map2.get(str(k).casefold())
                resolved_rename[kr or str(k)] = v
            rename_map = resolved_rename
            unknown = [k for k in rename_map.keys() if k not in col_set]
            if unknown:
                payload.setdefault("notes", [])
                payload["notes"].append(f"Ignored rename_columns for unknown columns: {', '.join(unknown[:10])}")
            applied: Dict[str, str] = {}
            for old, new in rename_map.items():
                if old not in col_set:
                    continue
                if new in col_set and new != old:
                    payload.setdefault("notes", [])
                    payload["notes"].append(f"Skipped rename '{old}' -> '{new}' (name conflict).")
                    continue
                applied[old] = new
            if applied:
                payload["columns"] = [applied.get(c, c) for c in out_columns]
                new_rows: List[Dict[str, Any]] = []
                for r in payload.get("rows") or []:
                    if not isinstance(r, dict):
                        continue
                    r2 = dict(r)
                    for old, new in applied.items():
                        if old in r2 and new != old:
                            r2[new] = r2.pop(old)
                    new_rows.append(r2)
                payload["rows"] = new_rows
                payload.setdefault("format", {})
                payload["format"]["renamed_columns"] = applied

    if spec2.derive:
        payload.setdefault("format", {})
        payload["format"]["derived_columns"] = [d.model_dump(mode="json") for d in (spec2.derive or [])]

    if include_summary:
        try:
            fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
            summ = build_format_summary_sv(
                spec=spec2,
                payload_format=fmt,
                derived_columns=(fmt.get("derived_columns") if isinstance(fmt.get("derived_columns"), list) else None),
                notes=(payload.get("notes") if isinstance(payload.get("notes"), list) else None),
            )
            payload.setdefault("format", {})
            payload["format"]["summary_sv"] = summ.get("summary_sv")
            payload["format"]["steps_sv"] = summ.get("steps_sv")
        except Exception:
            pass

    # Tag totals rows and replace marker with a human label for display.
    try:
        row_tags: List[List[str]] = []
        cols_for_totals = totals_columns or list(dims)
        for r in payload.get("rows") or []:
            if not isinstance(r, dict):
                row_tags.append([])
                continue
            is_tot = _is_total_row(r, dims=dims, totals_marker=totals_marker, totals_columns=cols_for_totals)
            row_tags.append(["total"] if is_tot else [])
            if is_tot and cols_for_totals:
                for c in cols_for_totals:
                    if str(r.get(c) or "") == totals_marker:
                        r[c] = totals_label
        payload.setdefault("format", {})
        payload["format"]["row_tags"] = row_tags
    except Exception:
        pass

    if table_key:
        payload["table_key"] = str(table_key)
    return payload, spec2, len(table_rows)


def build_presentation_table_payload_from_tool_run(
    *,
    source_tool_run_id: str,
    spec: FormatSpec,
    created_mode: str = "auto_default",
    title: Optional[str] = None,
    artifact_session_id: Optional[str] = None,
    artifact_turn_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Deterministic formatter:
    - Loads raw source from tool_runs.response_json via source_tool_run_id (no re-querying tools).
    - Applies: include_totals, unit conversion, rounding, sort, top_n, truncation bounds.
    - Produces UI-neutral payload.
    """
    res = (
        supabase
        .table("tool_runs")
        .select("id,session_id,turn_id,tool_name,response_json")
        .eq("id", source_tool_run_id)
        .limit(1)
        .execute()
    )
    rows = _res_data(res)
    if not rows:
        raise RuntimeError(f"tool_run not found: {source_tool_run_id}")
    tr = rows[0]

    # Session safety: never allow cross-session lineage.
    src_session_id = str(tr.get("session_id"))
    if artifact_session_id is not None and str(artifact_session_id) != src_session_id:
        raise ValueError(
            f"artifact_session_id mismatch: source tool_run session_id={src_session_id} "
            f"but artifact_session_id={artifact_session_id}"
        )

    response_json = tr.get("response_json") or {}
    if not isinstance(response_json, dict):
        raise ValueError("tool_runs.response_json must be an object/dict for formatting")

    meta_raw = response_json.get("meta") if isinstance(response_json.get("meta"), dict) else {}
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    tool_name = tr.get("tool_name")
    if tool_name is not None:
        meta.setdefault("_tool_name", str(tool_name))
    explicit_cols = response_json.get("columns") if isinstance(response_json.get("columns"), list) else None

    # Single-table response (income_statement_tool style)
    if isinstance(response_json.get("table"), list):
        payload, spec2, row_count = _build_single_presentation_table_payload(
            meta=meta,
            table_rows=[r for r in (response_json.get("table") or []) if isinstance(r, dict)],
            explicit_cols=[str(c) for c in explicit_cols] if explicit_cols else None,
            spec=spec,
            include_summary=True,
            table_key=None,
        )
        payload.setdefault("format", {})
        payload["format"]["presentation_kind"] = "single_table"
        payload["format"]["tables_count"] = 1
    else:
        # Multi-table response (variance_tool style): format ALL list[dict] tables into ONE artifact.
        table_keys: List[str] = []
        tables: List[Dict[str, Any]] = []
        total_rows = 0
        for k, v in response_json.items():
            if k in {"meta", "columns", "table"}:
                continue
            # Skip debug/internal tables (e.g. variance_tool returns "_meta")
            if str(k).startswith("_"):
                continue
            if isinstance(v, list) and (not v or all(isinstance(x, dict) for x in v)):
                table_keys.append(str(k))
                p, _spec2_local, rc = _build_single_presentation_table_payload(
                    meta=meta,
                    table_rows=[x for x in v if isinstance(x, dict)],
                    explicit_cols=None,
                    spec=spec,
                    include_summary=False,  # show summary once at top-level
                    table_key=str(k),
                )
                total_rows += int(rc)
                tables.append(p)
        if not tables:
            raise ValueError("No table found in tool_runs.response_json (expected 'table' or list-of-dicts tables).")

        # Use the first table to store the effective spec (for incremental formatting & UI defaults)
        _first_payload, spec2, _ = _build_single_presentation_table_payload(
            meta=meta,
            table_rows=[x for x in (response_json.get(table_keys[0]) or []) if isinstance(x, dict)],
            explicit_cols=None,
            spec=spec,
            include_summary=False,
            table_key=table_keys[0],
        )

        payload = {
            "kind": "multi_table",
            "tables": tables,
            "format": {
                "unit": _DISPLAY_UNIT.get(spec2.unit, spec2.unit),
                "unit_canonical": spec2.unit,
                "decimals": spec2.decimals,
                "presentation_kind": "multi_table",
                "tables_count": len(tables),
            },
        }
        row_count = int(total_rows)
        try:
            summ = build_format_summary_sv(spec=spec2, payload_format=payload.get("format"), derived_columns=None, notes=None)
            payload.setdefault("format", {})
            payload["format"]["summary_sv"] = summ.get("summary_sv")
            payload["format"]["steps_sv"] = summ.get("steps_sv")
        except Exception:
            pass

    artifact_row: Dict[str, Any] = {
        # Important: allow reformat artifacts to be created on a different turn than the source tool run.
        "session_id": str(artifact_session_id) if artifact_session_id is not None else src_session_id,
        "turn_id": int(artifact_turn_id) if artifact_turn_id is not None else int(tr.get("turn_id")),
        "artifact_type": "presentation_table",
        "title": title or "Presentation",
        "created_mode": created_mode,
        "source_tool_run_id": tr.get("id"),
        "source_tool_name": tr.get("tool_name"),
        "format_spec": spec2.model_dump(mode="json"),
        "payload": payload,
        "row_count": int(row_count),
        "bytes": _estimate_bytes(payload),
    }
    return artifact_row


def insert_presentation_artifact(artifact_row: Dict[str, Any]) -> str:
    res = supabase.table("artifacts").insert(artifact_row).execute()
    err = _res_error(res)
    if err:
        raise RuntimeError(f"Insert artifacts failed: {err}")
    rows = _res_data(res)
    if not rows:
        raise RuntimeError("Insert artifacts returned no rows.")
    return str(rows[0].get("id") or "")


def _strip_history_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prevent recursive nesting of history inside history.
    Only store the presentation payload, not its undo/redo stacks.
    """
    out = dict(payload or {})
    out.pop("lineage", None)
    out.pop("redo_lineage", None)
    return out


def _fetch_singleton_presentation_artifact_row(
    session_id: str,
    turn_id: int,
    *,
    source_tool_name: str,
) -> Optional[Dict[str, Any]]:
    res = (
        supabase
        .table("artifacts")
        .select("id,created_at,updated_at,format_spec,source_tool_run_id,source_tool_name,title,created_mode,payload,row_count,bytes")
        .eq("session_id", str(session_id))
        .eq("turn_id", int(turn_id))
        .eq("artifact_type", "presentation_table")
        .eq("source_tool_name", str(source_tool_name))
        .order("updated_at", desc=True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = _res_data(res)
    if rows:
        return rows[0]
    return None


def _snapshot_from_artifact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    return {
        "ts": time.time(),
        "artifact_id": row.get("id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "title": row.get("title"),
        "created_mode": row.get("created_mode"),
        "source_tool_run_id": row.get("source_tool_run_id"),
        "source_tool_name": row.get("source_tool_name"),
        "format_spec": row.get("format_spec"),
        "payload": _strip_history_payload(payload),
        "row_count": row.get("row_count"),
        "bytes": row.get("bytes"),
    }


def undo_singleton_presentation_artifact(
    *,
    session_id: str,
    turn_id: int,
    source_tool_name: str,
    history_limit: int = 10,
    redo_limit: int = 10,
) -> Dict[str, Any]:
    """
    Undo formatting by switching the active singleton artifact to the last snapshot in payload.lineage.
    Persists redo history in payload.redo_lineage.
    """
    row = _fetch_singleton_presentation_artifact_row(
        session_id=session_id,
        turn_id=turn_id,
        source_tool_name=str(source_tool_name),
    )
    if not row:
        raise RuntimeError("No presentation_table artifact found for this turn + tool.")

    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), list) else []
    redo = payload.get("redo_lineage") if isinstance(payload.get("redo_lineage"), list) else []
    lineage = [x for x in lineage if isinstance(x, dict)]
    redo = [x for x in redo if isinstance(x, dict)]

    if not lineage:
        raise RuntimeError("Nothing to undo (lineage is empty).")

    prev = lineage.pop()
    if not isinstance(prev.get("payload"), dict):
        raise RuntimeError("Cannot undo: lineage entry missing payload snapshot (older history format).")

    redo.append(_snapshot_from_artifact_row(row))
    if redo_limit > 0:
        redo = redo[-int(redo_limit):]

    restored_payload = dict(prev.get("payload") or {})
    restored_payload["lineage"] = lineage[-int(history_limit):] if history_limit > 0 else lineage
    restored_payload["redo_lineage"] = redo

    update_payload = {
        "title": prev.get("title") or row.get("title"),
        "created_mode": "undo",
        "source_tool_run_id": prev.get("source_tool_run_id") or row.get("source_tool_run_id"),
        "source_tool_name": prev.get("source_tool_name") or row.get("source_tool_name"),
        "format_spec": prev.get("format_spec") or row.get("format_spec"),
        "payload": restored_payload,
        "row_count": prev.get("row_count"),
        "bytes": prev.get("bytes"),
        "parent_artifact_id": None,
    }
    res_upd = supabase.table("artifacts").update(update_payload).eq("id", str(row.get("id"))).execute()
    _ = _res_data(res_upd)
    return {
        "artifact_id": str(row.get("id")),
        "mode": "undo",
        "undo_depth": len(restored_payload.get("lineage") or []),
        "redo_depth": len(restored_payload.get("redo_lineage") or []),
    }


def redo_singleton_presentation_artifact(
    *,
    session_id: str,
    turn_id: int,
    source_tool_name: str,
    history_limit: int = 10,
    redo_limit: int = 10,
) -> Dict[str, Any]:
    """
    Redo formatting by switching the active singleton artifact to the last snapshot in payload.redo_lineage.
    """
    row = _fetch_singleton_presentation_artifact_row(
        session_id=session_id,
        turn_id=turn_id,
        source_tool_name=str(source_tool_name),
    )
    if not row:
        raise RuntimeError("No presentation_table artifact found for this turn + tool.")

    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), list) else []
    redo = payload.get("redo_lineage") if isinstance(payload.get("redo_lineage"), list) else []
    lineage = [x for x in lineage if isinstance(x, dict)]
    redo = [x for x in redo if isinstance(x, dict)]

    if not redo:
        raise RuntimeError("Nothing to redo (redo_lineage is empty).")

    nxt = redo.pop()
    if not isinstance(nxt.get("payload"), dict):
        raise RuntimeError("Cannot redo: redo entry missing payload snapshot.")

    lineage.append(_snapshot_from_artifact_row(row))
    if history_limit > 0:
        lineage = lineage[-int(history_limit):]

    restored_payload = dict(nxt.get("payload") or {})
    restored_payload["lineage"] = lineage
    restored_payload["redo_lineage"] = redo[-int(redo_limit):] if redo_limit > 0 else redo

    update_payload = {
        "title": nxt.get("title") or row.get("title"),
        "created_mode": "redo",
        "source_tool_run_id": nxt.get("source_tool_run_id") or row.get("source_tool_run_id"),
        "source_tool_name": nxt.get("source_tool_name") or row.get("source_tool_name"),
        "format_spec": nxt.get("format_spec") or row.get("format_spec"),
        "payload": restored_payload,
        "row_count": nxt.get("row_count"),
        "bytes": nxt.get("bytes"),
        "parent_artifact_id": None,
    }
    res_upd = supabase.table("artifacts").update(update_payload).eq("id", str(row.get("id"))).execute()
    _ = _res_data(res_upd)
    return {
        "artifact_id": str(row.get("id")),
        "mode": "redo",
        "undo_depth": len(restored_payload.get("lineage") or []),
        "redo_depth": len(restored_payload.get("redo_lineage") or []),
    }


def upsert_singleton_presentation_artifact(
    *,
    session_id: str,
    turn_id: int,
    artifact_row: Dict[str, Any],
    history_limit: int = 10,
) -> Dict[str, Any]:
    """
    Ensure exactly ONE presentation_table artifact exists per (session_id, turn_id, source_tool_name).

    Behavior:
    - If none exists: insert and return {"artifact_id": ..., "changed": True}
    - If one exists and the requested (source_tool_run_id + format_spec) matches: return it (no write)
    - If one exists and differs: overwrite it (UPDATE in place) but store previous versions as full snapshots
      in payload.lineage (undo stack). Also clears payload.redo_lineage (redo stack) because a new change
      invalidates redo history.
    """
    session_id = str(session_id)
    turn_id_i = int(turn_id)
    source_tool_name = str(artifact_row.get("source_tool_name") or "").strip()
    if not source_tool_name:
        raise ValueError("upsert_singleton_presentation_artifact requires artifact_row['source_tool_name'].")

    # Fetch existing singleton (if any)
    res = (
        supabase
        .table("artifacts")
        .select("id,created_at,updated_at,format_spec,source_tool_run_id,source_tool_name,title,created_mode,payload")
        .eq("session_id", session_id)
        .eq("turn_id", turn_id_i)
        .eq("artifact_type", "presentation_table")
        .eq("source_tool_name", source_tool_name)
        .order("updated_at", desc=True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    existing_rows = _res_data(res)
    if not existing_rows:
        try:
            new_id = insert_presentation_artifact(artifact_row)
            return {"artifact_id": new_id, "changed": True, "mode": "insert"}
        except Exception as e:
            # If a DB unique index exists, a race could cause a conflict. Retry as overwrite.
            msg = str(e).lower()
            if "duplicate" in msg or "unique" in msg:
                # Re-fetch and continue through overwrite path
                res2 = (
                    supabase
                    .table("artifacts")
                    .select("id,created_at,updated_at,format_spec,source_tool_run_id,source_tool_name,title,created_mode,payload")
                    .eq("session_id", session_id)
                    .eq("turn_id", turn_id_i)
                    .eq("artifact_type", "presentation_table")
                    .eq("source_tool_name", source_tool_name)
                    .order("updated_at", desc=True)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                existing_rows2 = _res_data(res2)
                if not existing_rows2:
                    raise
                ex = existing_rows2[0]
                ex_id = str(ex.get("id") or "")
                artifact_row["payload"] = artifact_row.get("payload") or {}
                # Fall through to overwrite by setting up locals
                existing_rows = [ex]  # type: ignore[assignment]
            else:
                raise

    ex = existing_rows[0]
    ex_id = str(ex.get("id") or "")
    ex_spec = ex.get("format_spec")
    ex_src = ex.get("source_tool_run_id")
    ex_payload = ex.get("payload") if isinstance(ex.get("payload"), dict) else {}

    new_spec = artifact_row.get("format_spec")
    new_src = artifact_row.get("source_tool_run_id")
    new_payload = artifact_row.get("payload") if isinstance(artifact_row.get("payload"), dict) else {}

    # If effectively identical, return existing (no duplicate rows, no updates)
    try:
        ex_spec_norm = json.dumps(ex_spec, ensure_ascii=False, sort_keys=True)
        new_spec_norm = json.dumps(new_spec, ensure_ascii=False, sort_keys=True)
    except Exception:
        ex_spec_norm = str(ex_spec)
        new_spec_norm = str(new_spec)

    # Notes are user-visible feedback; even if spec+source are unchanged, we may still want to update notes.
    def _norm_notes(p: Dict[str, Any]) -> str:
        n = p.get("notes")
        if not isinstance(n, list):
            return "[]"
        n2 = [str(x).strip() for x in n if isinstance(x, str) and str(x).strip()]
        try:
            return json.dumps(n2, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(n2)

    notes_changed = _norm_notes(ex_payload) != _norm_notes(new_payload)

    # Compare payload too (excluding history + notes). This avoids getting "stuck" when:
    # - formatting implementation changes
    # - source data changes (same tool_run_id but recalculated artifact)
    # - tool meta changes that affect presentation
    def _norm_payload_wo_notes(p: Dict[str, Any]) -> str:
        p2 = _strip_history_payload(p)
        if isinstance(p2, dict):
            p2.pop("notes", None)
        try:
            return json.dumps(p2, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(p2)

    payload_changed = _norm_payload_wo_notes(ex_payload) != _norm_payload_wo_notes(new_payload)

    if str(ex_src) == str(new_src) and ex_spec_norm == new_spec_norm and (not notes_changed) and (not payload_changed):
        return {"artifact_id": ex_id, "changed": False, "mode": "unchanged"}
    if str(ex_src) == str(new_src) and ex_spec_norm == new_spec_norm and notes_changed and (not payload_changed):
        # Update notes in-place WITHOUT adding lineage (since the rendering payload is otherwise identical).
        update_payload = {
            "title": artifact_row.get("title"),
            "created_mode": artifact_row.get("created_mode"),
            "payload": new_payload,
            "bytes": artifact_row.get("bytes"),
        }
        res_upd = supabase.table("artifacts").update(update_payload).eq("id", ex_id).execute()
        _ = _res_data(res_upd)
        return {"artifact_id": ex_id, "changed": True, "mode": "notes_update"}

    # Build lineage history inside payload (bounded). This acts as an undo stack.
    prev_entry = {
        "ts": time.time(),
        "artifact_id": ex_id,
        "created_at": ex.get("created_at"),
        "updated_at": ex.get("updated_at"),
        "title": ex.get("title"),
        "created_mode": ex.get("created_mode"),
        "source_tool_run_id": ex_src,
        "source_tool_name": ex.get("source_tool_name"),
        "format_spec": ex_spec,
        "payload": _strip_history_payload(ex_payload),
        "row_count": ex.get("row_count"),
        "bytes": ex.get("bytes"),
    }

    new_payload = artifact_row.get("payload") if isinstance(artifact_row.get("payload"), dict) else {}
    ex_payload = ex.get("payload") if isinstance(ex.get("payload"), dict) else {}
    lineage: List[Dict[str, Any]] = []
    if isinstance(ex_payload.get("lineage"), list):
        lineage = [x for x in ex_payload.get("lineage") if isinstance(x, dict)]
    lineage.append(prev_entry)
    if history_limit > 0:
        lineage = lineage[-int(history_limit):]
    if isinstance(new_payload, dict):
        new_payload["lineage"] = lineage
        # Any new overwrite invalidates redo.
        new_payload["redo_lineage"] = []
        artifact_row["payload"] = new_payload

    # Overwrite existing row in-place
    update_payload = {
        "title": artifact_row.get("title"),
        "created_mode": artifact_row.get("created_mode"),
        "source_tool_run_id": artifact_row.get("source_tool_run_id"),
        "source_tool_name": artifact_row.get("source_tool_name"),
        "format_spec": artifact_row.get("format_spec"),
        "payload": artifact_row.get("payload"),
        "row_count": artifact_row.get("row_count"),
        "bytes": artifact_row.get("bytes"),
        "parent_artifact_id": None,  # singleton mode; history is in payload.lineage
    }
    res_upd = supabase.table("artifacts").update(update_payload).eq("id", ex_id).execute()
    _ = _res_data(res_upd)  # ensure request executed
    return {"artifact_id": ex_id, "changed": True, "mode": "overwrite"}


