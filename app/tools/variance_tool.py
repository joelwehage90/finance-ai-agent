from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Viktigt: använd den gemensamma Supabase-klienten som byggs från .env via settings.py
from ..supabase_client import supabase
from .metadata import get_relation_docs, pick_column_docs


FilterValue = Union[str, List[str]]


@dataclass(frozen=True)
class PeriodRange:
    start: date          # inkl
    end_excl: date       # exkl (praktiskt för gte/lt)


def _parse_yyyymm(period: str) -> pd.Period:
    """Parse 'YYYY-MM' till pandas Period (M)."""
    return pd.Period(period, freq="M")


def _month_range(period: str) -> PeriodRange:
    """En månad: [månadens första dag, nästa månads första dag)."""
    p = _parse_yyyymm(period)
    start = p.start_time.date()
    end_excl = (p + 1).start_time.date()
    return PeriodRange(start=start, end_excl=end_excl)


def _ytd_range(period: str) -> PeriodRange:
    """YTD: [1 jan samma år, första dagen efter period-månaden)."""
    p = _parse_yyyymm(period)
    start = date(p.year, 1, 1)
    end_excl = (p + 1).start_time.date()
    return PeriodRange(start=start, end_excl=end_excl)


def _apply_filters(q, filters: Optional[Dict[str, FilterValue]]) -> Any:
    """Applicera eq / in_ filter (OR inom dimension via lista)."""
    if not filters:
        return q
    for col, val in filters.items():
        if val is None:
            continue
        if isinstance(val, list):
            q = q.in_(col, val)
        else:
            q = q.eq(col, val)
    return q


def _fetch_all_rows(
    *,
    table_name: str,
    select_cols: List[str],
    date_col: str,
    dr: PeriodRange,
    filters: Optional[Dict[str, FilterValue]],
    page_size: int = 1000,
    order_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Robust pagination för Supabase/PostgREST:
    - Stabil order() så offset-pagination inte tappar/duplicerar rader
    - offset stegas med len(rows)
    - bryt först när servern returnerar 0 rader
    """
    if page_size <= 0:
        raise ValueError("page_size måste vara > 0")

    out: List[Dict[str, Any]] = []
    offset = 0

    def _as_date(v: Any) -> Optional[date]:
        """
        Best-effort: normalize a PostgREST-returned date/timestamp/string to a date.
        Returns None if parsing fails.
        """
        if v is None:
            return None
        if isinstance(v, date) and not isinstance(v, pd.Timestamp):
            # includes plain datetime.date (but NOT pandas Timestamp)
            return v
        # datetime (or pandas timestamp) -> date
        try:
            if hasattr(v, "date") and callable(v.date):
                return v.date()
        except Exception:
            pass
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            # ISO date or timestamp; keep only YYYY-MM-DD
            try:
                return date.fromisoformat(s[:10])
            except Exception:
                return None
        return None

    # Default: sortera deterministiskt på date_col + övriga kolumner i select (om inget angivet)
    if order_cols is None:
        order_cols = [date_col] + [c for c in select_cols if c != date_col]

    while True:
        q = supabase.table(table_name).select(",".join(select_cols))
        # IMPORTANT (PostgREST/supabase-py quirk):
        # Many client implementations store filters in a dict keyed by column name, which means
        # applying two filters on the same column (gte + lt) can cause one to overwrite the other.
        # To avoid silently dropping the lower/upper bound, we apply ONLY the lower bound in SQL
        # and enforce end_excl locally (and stop paginating once we reach it).
        q = q.gte(date_col, str(dr.start))
        q = _apply_filters(q, filters)

        for c in order_cols:
            q = q.order(c, desc=False)

        resp = q.range(offset, offset + page_size - 1).execute()
        rows = resp.data or []

        if not rows:
            break

        # Enforce end_excl locally; because we order by date_col asc, we can stop early.
        in_range: List[Dict[str, Any]] = []
        hit_end = False
        for r in rows:
            rv = r.get(date_col) if isinstance(r, dict) else None
            d = _as_date(rv)
            if d is not None and d >= dr.end_excl:
                hit_end = True
                continue
            in_range.append(r)

        out.extend(in_range)
        offset += len(rows)
        if hit_end:
            break

    return out


def _choose_source_view(
    grain: List[str],
    filters: Optional[Dict[str, FilterValue]],
) -> Tuple[str, str]:
    """
    Välj vy beroende på vilka dimensioner som efterfrågas.
    - pnl_monthly_rr: rr_level_1, rr_level_2, unit, konto_typ, amount, month_start
    - transactions_view: har mer (account, supplier, product, project, ...)
    """
    pnl_cols = {"month_start", "rr_level_1", "rr_level_2", "unit", "konto_typ", "amount"}
    needed = set(grain) | {"konto_typ", "amount"}
    if filters:
        needed |= set(filters.keys())

    if needed.issubset(pnl_cols):
        return ("pnl_monthly_rr", "month_start")
    return ("transactions_view", "date")


def _aggregate_amount(df: pd.DataFrame, grain: List[str]) -> pd.DataFrame:
    """Groupby grain + konto_typ och summera amount."""
    group_cols = list(grain) + ["konto_typ"]
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["amount"])
    return df.groupby(group_cols, dropna=False, as_index=False)["amount"].sum()


def _build_variance(agg_base: pd.DataFrame, agg_comp: pd.DataFrame, grain: List[str]) -> pd.DataFrame:
    """Outer-merge och beräkna delta."""
    key_cols = list(grain) + ["konto_typ"]

    base = agg_base.rename(columns={"amount": "amount_base"})
    comp = agg_comp.rename(columns={"amount": "amount_comp"})

    merged = base.merge(comp, on=key_cols, how="outer")
    merged["amount_base"] = merged["amount_base"].fillna(0)
    merged["amount_comp"] = merged["amount_comp"].fillna(0)
    merged["delta"] = merged["amount_comp"] - merged["amount_base"]
    return merged


def _add_explain_cols(df: pd.DataFrame, *, is_negative_table: bool) -> pd.DataFrame:
    """
    explain_pct och explain_pct_cum inom tabellen.
    - Positiv tabell: delta / sum(delta)
    - Negativ tabell: (-delta) / sum(-delta)  (så procent blir positiv)
    Sortering görs så "störst påverkan" kommer först.
    """
    if df.empty:
        df["explain_pct"] = pd.Series(dtype="float64")
        df["explain_pct_cum"] = pd.Series(dtype="float64")
        return df

    if is_negative_table:
        weights = (-df["delta"]).clip(lower=0)
        denom = weights.sum()
        df = df.sort_values("delta", ascending=True)  # mest negativ först
        df["explain_pct"] = (weights / denom) if denom != 0 else 0.0
    else:
        weights = df["delta"].clip(lower=0)
        denom = weights.sum()
        df = df.sort_values("delta", ascending=False)
        df["explain_pct"] = (weights / denom) if denom != 0 else 0.0

    df["explain_pct_cum"] = df["explain_pct"].cumsum()
    return df


def _add_relation_docs_to_meta(
    meta: Dict[str, Any],
    *,
    schema: str,
    relation: str,
    used_source_cols: List[str],
    warnings: List[str],
) -> None:
    """
    Läser COMMENT ON (relation + kolumner) och lägger in i meta.
    Om RPC saknas/behörighet saknas -> lägger warning men kraschar inte tool.
    """
    meta.setdefault("source", {"schema": schema, "relation": relation})

    try:
        docs = get_relation_docs(schema, relation)
        meta["docs"] = {
            "relation_comment": docs.get("relation_comment"),
            "columns": pick_column_docs(docs, used_source_cols),
        }
    except Exception as e:
        warnings.append(f"Could not load relation docs for {schema}.{relation}: {type(e).__name__}: {e}")


def variance_tables(
    *,
    compare_mode: str,  # "month" | "ytd"
    base_period: str,   # "YYYY-MM"
    comp_period: str,   # "YYYY-MM"
    grain: List[str],
    filters: Optional[Dict[str, FilterValue]] = None,
    top_n_pos: Optional[int] = 50,
    top_n_neg: Optional[int] = 50,
) -> Dict[str, Any]:
    """
    Hämtar data från Supabase och returnerar fyra tabeller:
      - kostnader_pos (delta > 0)
      - kostnader_neg (delta < 0)
      - intakter_pos  (delta > 0)
      - intakter_neg  (delta < 0)

    Varje tabell innehåller:
      grain-kolumner, amount_base, amount_comp, delta, explain_pct, explain_pct_cum

    Returnerar dessutom:
      - meta: dict (inkl. COMMENT ON docs + semantiska hints)
      - _meta: DataFrame (bakåtkompatibilitet)
    """
    compare_mode = compare_mode.lower().strip()
    if compare_mode not in {"month", "ytd"}:
        raise ValueError("compare_mode måste vara 'month' eller 'ytd'.")

    if not grain:
        raise ValueError("grain måste innehålla minst en dimension (t.ex. ['rr_level_2']).")

    # Normalize grain:
    # - strip + de-dupe (stable)
    # - do NOT allow 'konto_typ' inside grain, because the tool already splits output tables by konto_typ
    #   and always includes it internally for grouping.
    grain_norm: List[str] = []
    seen = set()
    for c in grain:
        c2 = str(c).strip()
        if not c2:
            continue
        if c2 == "konto_typ":
            continue
        if c2 in seen:
            continue
        seen.add(c2)
        grain_norm.append(c2)
    grain = grain_norm
    if not grain:
        raise ValueError("grain får inte bara vara ['konto_typ']; välj en annan dimension (t.ex. ['rr_level_2']).")

    # Period -> datumintervall
    base_dr = _month_range(base_period) if compare_mode == "month" else _ytd_range(base_period)
    comp_dr = _month_range(comp_period) if compare_mode == "month" else _ytd_range(comp_period)

    # Välj datakälla
    table_name, date_col = _choose_source_view(grain, filters)

    # Välj kolumner (inkl date_col för enkel debugging)
    select_cols = sorted(set(grain + ["konto_typ", "amount", date_col]))

    warnings: List[str] = []

    # Meta (agentvänlig)
    meta: Dict[str, Any] = {
        "compare_mode": compare_mode,
        "base_period": base_period,
        "comp_period": comp_period,
        "grain": grain,
        "filters": filters or {},
        "top_n_pos": top_n_pos,
        "top_n_neg": top_n_neg,
        "source_view": table_name,
        "date_col": date_col,
        "base_start": str(base_dr.start),
        "base_end_excl": str(base_dr.end_excl),
        "comp_start": str(comp_dr.start),
        "comp_end_excl": str(comp_dr.end_excl),
        "semantic_hints": {
            "amount_sign": "revenue_positive_cost_negative",
        },
        # Formatting hints (UI-neutral): help the formatting layer classify columns deterministically.
        # - rows: dimension columns for presentation (dims)
        # - value_columns: numeric columns
        # - column_formats: per-column render hints (Lovable/any UI can use this)
        # Each returned table is already split by konto_typ (kostnader/intäkter), so we don't repeat it as a column.
        "rows": list(grain),
        # Use period names as the primary value columns (more readable than amount_base/amount_comp)
        "value_columns": [base_period, comp_period, "delta", "explain_pct", "explain_pct_cum"],
        "column_formats": {
            "explain_pct": {"kind": "percent", "scale": "fraction", "decimals": 1},
            "explain_pct_cum": {"kind": "percent", "scale": "fraction", "decimals": 1},
        },
        # Tool-specific default sort: explain share (largest drivers first) unless user overrides.
        "default_sort": [{"col": "explain_pct", "dir": "desc"}],
        "warnings": warnings,
    }

    # Hämta COMMENT ON docs för källvyn + relevanta kolumner
    used_source_cols = sorted(set(select_cols + (list(filters.keys()) if filters else [])))
    _add_relation_docs_to_meta(
        meta,
        schema="public",
        relation=table_name,
        used_source_cols=used_source_cols,
        warnings=warnings,
    )

    # Hämta + aggreggera base och comp
    rows_base = _fetch_all_rows(
        table_name=table_name,
        select_cols=select_cols,
        date_col=date_col,
        dr=base_dr,
        filters=filters,
    )
    rows_comp = _fetch_all_rows(
        table_name=table_name,
        select_cols=select_cols,
        date_col=date_col,
        dr=comp_dr,
        filters=filters,
    )

    df_base = pd.DataFrame(rows_base)
    df_comp = pd.DataFrame(rows_comp)

    # Säkerställ numeriskt
    for df in (df_base, df_comp):
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    agg_base = _aggregate_amount(df_base, grain)
    agg_comp = _aggregate_amount(df_comp, grain)

    merged = _build_variance(agg_base, agg_comp, grain)

    # Konto_typ i din modell
    KONTO_KOST = "kostnader"
    KONTO_INT = "intäkter"

    def _make_table(konto_typ: str, sign: str) -> pd.DataFrame:
        sub = merged[merged["konto_typ"] == konto_typ].copy()

        if sign == "pos":
            sub = sub[sub["delta"] > 0].copy()
            sub = _add_explain_cols(sub, is_negative_table=False)
            if top_n_pos is not None:
                sub = sub.head(top_n_pos)
        else:
            sub = sub[sub["delta"] < 0].copy()
            sub = _add_explain_cols(sub, is_negative_table=True)
            if top_n_neg is not None:
                sub = sub.head(top_n_neg)

        cols = grain + ["amount_base", "amount_comp", "delta", "explain_pct", "explain_pct_cum"]

        # Om någon grain-kolumn saknas (tomt dataset), skapa den
        for c in grain:
            if c not in sub.columns:
                sub[c] = pd.Series(dtype="object")

        out_df = sub[cols].reset_index(drop=True)
        # Rename base/comp columns to period names for readability in presentation artifacts/UI.
        out_df = out_df.rename(columns={"amount_base": base_period, "amount_comp": comp_period})
        return out_df

    out = {
        "meta": meta,
        "kostnader_pos": _make_table(KONTO_KOST, "pos"),
        "kostnader_neg": _make_table(KONTO_KOST, "neg"),
        "intakter_pos":  _make_table(KONTO_INT,  "pos"),
        "intakter_neg":  _make_table(KONTO_INT,  "neg"),
        # Behåller din gamla _meta som snabb, platt debug
        "_meta": pd.DataFrame([{
            "source_view": table_name,
            "date_col": date_col,
            "compare_mode": compare_mode,
            "base_period": base_period,
            "comp_period": comp_period,
            "base_start": str(base_dr.start),
            "base_end_excl": str(base_dr.end_excl),
            "comp_start": str(comp_dr.start),
            "comp_end_excl": str(comp_dr.end_excl),
        }]),
    }

    return out