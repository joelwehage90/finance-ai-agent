# %% [Cell 1] Imports + Supabase-klient
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..supabase_client import supabase
from .metadata import get_relation_docs, pick_column_docs


FilterValue = Union[str, List[str]]


# %% [Cell 2] Hjälpfunktioner: perioder, vyval, filter, pagination-hämtning

@dataclass(frozen=True)
class PeriodRange:
    start: date
    end_excl: date  # exkl (praktiskt för gte/lt)


def _parse_yyyymm(period: str) -> pd.Period:
    return pd.Period(period, freq="M")


def _month_range(period: str) -> PeriodRange:
    p = _parse_yyyymm(period)
    return PeriodRange(start=p.start_time.date(), end_excl=(p + 1).start_time.date())


def _ytd_range(period: str) -> PeriodRange:
    p = _parse_yyyymm(period)
    return PeriodRange(start=date(p.year, 1, 1), end_excl=(p + 1).start_time.date())


def _overall_range(compare_mode: str, periods: List[str]) -> PeriodRange:
    """
    Min/Max datumintervall som täcker alla perioder.
    - month: från min period start till (max period + 1 månad)
    - ytd: från 1 jan min-år till (max period + 1 månad)
    """
    ps = [_parse_yyyymm(p) for p in periods]
    p_min, p_max = min(ps), max(ps)
    end_excl = (p_max + 1).start_time.date()
    if compare_mode == "month":
        start = p_min.start_time.date()
    else:
        start = date(p_min.year, 1, 1)
    return PeriodRange(start=start, end_excl=end_excl)


def _apply_filters(q, filters: Optional[Dict[str, FilterValue]]) -> Any:
    """eq / in_ (OR inom dimension via lista)."""
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
    Hämta ALLA rader med pagination.

    Robust version:
    - Kör med page_size <= serverns max (default 1000)
    - Stabil order() så offset-pagination inte tappar/duplicerar
    - offset stegas med len(rows)
    - stoppar när servern returnerar 0 rader
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
            return v
        try:
            if hasattr(v, "date") and callable(v.date):
                return v.date()
        except Exception:
            pass
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return date.fromisoformat(s[:10])
            except Exception:
                return None
        return None

    # Välj en deterministisk sorteringsordning.
    if order_cols is None:
        preferred = [
            date_col,
            "konto_typ",
            "rr_level_1",
            "rr_level_2",
            "unit",
            "account",
            "supplier",
            "product",
            "project",
            "amount",
        ]
        select_set = set(select_cols)

        order_cols = []
        seen = set()

        # 1) preferred (bara om kolumnen finns i select eller är date_col)
        for c in preferred:
            if c == date_col or c in select_set:
                if c not in seen:
                    order_cols.append(c)
                    seen.add(c)

        # 2) resten av select_cols
        for c in select_cols:
            if c not in seen:
                order_cols.append(c)
                seen.add(c)

    while True:
        q = supabase.table(table_name).select(",".join(select_cols))
        # IMPORTANT (PostgREST/supabase-py quirk):
        # Some clients overwrite same-column filters, so gte+lt can silently drop one bound.
        # Apply only the lower bound in SQL and enforce end_excl locally (stop early once reached).
        q = q.gte(date_col, str(dr.start))
        q = _apply_filters(q, filters)

        # Stabil ordering för offset/range
        for c in order_cols:
            q = q.order(c, desc=False)

        resp = q.range(offset, offset + page_size - 1).execute()
        rows = resp.data or []

        if not rows:
            break

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


def _choose_source_view(rows: List[str], filters: Optional[Dict[str, FilterValue]]) -> Tuple[str, str]:
    """
    Välj snabb vy om möjligt:
    pnl_monthly_rr har: month_start, rr_level_1, rr_level_2, unit, konto_typ, amount
    annars transactions_view (date, amount, account, unit, supplier, product, project, rr_level_1, rr_level_2, konto_typ)
    """
    pnl_cols = {"month_start", "rr_level_1", "rr_level_2", "unit", "konto_typ", "amount"}
    needed = set(rows) | {"amount"}
    if filters:
        needed |= set(filters.keys())

    if needed.issubset(pnl_cols):
        return ("pnl_monthly_rr", "month_start")
    return ("transactions_view", "date")


def _normalize_dim_nulls(df: pd.DataFrame, dims: List[str]) -> pd.DataFrame:
    """Fyll null i dimensionskolumner så groupby/pivot blir stabila."""
    for c in dims:
        if c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(-1)
            else:
                df[c] = df[c].fillna("(null)")
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


# %% [Cell 3] P&L tool: pnl_tables(...)

def pnl_tables(
    *,
    compare_mode: str,                 # "month" | "ytd"
    periods: List[str],                # ["YYYY-MM", ...]
    rows: List[str],                   # dimensions for "rader"
    filters: Optional[Dict[str, FilterValue]] = None,
    include_total: bool = True,
) -> Dict[str, Any]:
    """
    Returnerar en "pivot"-vänlig P&L-tabell i wide-format:
      - Kolumner: rows-dimensioner + en kolumn per period
      - Värde: amount (month eller ytd)

    Metadata inkluderas: compare_mode, periods, rows, filters, datakälla (vy) + relation docs (COMMENT ON).
    """
    compare_mode = compare_mode.lower().strip()
    if compare_mode not in {"month", "ytd"}:
        raise ValueError("compare_mode måste vara 'month' eller 'ytd'.")

    if not periods:
        raise ValueError("periods får inte vara tom.")
    if not rows:
        raise ValueError("rows måste innehålla minst en dimension (t.ex. ['konto_typ','rr_level_1']).")

    # Välj vy + datumkolumn
    source_view, date_col = _choose_source_view(rows, filters)

    # Datumintervall som täcker allt vi behöver hämta
    dr = _overall_range(compare_mode, periods)

    # Kolumner att hämta (källkolumner)
    filter_cols = list(filters.keys()) if filters else []
    select_cols = sorted(set(rows + filter_cols + ["amount", date_col]))

    warnings: List[str] = []

    # Hämta data
    raw_rows = _fetch_all_rows(
        table_name=source_view,
        select_cols=select_cols,
        date_col=date_col,
        dr=dr,
        filters=filters,
    )
    df = pd.DataFrame(raw_rows)

    # Grund-meta (oavsett om df är tom)
    meta: Dict[str, Any] = {
        "compare_mode": compare_mode,
        "periods": periods,
        "rows": rows,
        "filters": filters or {},
        "source_view": source_view,
        "date_col": date_col,
        "fetched_start": str(dr.start),
        "fetched_end_excl": str(dr.end_excl),
        "format": "wide",
        "include_total": include_total,
        # Nya fält:
        "semantic_hints": {
            "amount_sign": "revenue_positive_cost_negative",
        },
        "warnings": warnings,
    }

    # Lägg in COMMENT ON docs i meta (för källkolumnerna)
    _add_relation_docs_to_meta(
        meta,
        schema="public",
        relation=source_view,
        used_source_cols=select_cols,
        warnings=warnings,
    )

    # Om inget data: returnera tom pivot med rätt kolumner
    if df.empty:
        empty = pd.DataFrame(columns=rows + periods)
        return {
            "meta": meta,
            "table": empty,
        }

    # Validera att dimensioner finns
    missing = [c for c in (rows + filter_cols) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Följande kolumner saknas i datan från {source_view}: {missing}. "
            f"Kontrollera rows/filters och att vyn innehåller kolumnerna."
        )

    # Normalisera typer
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Parse datum
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Om allt föll bort i datum-parse: returnera tomt
    if df.empty:
        warnings.append("All rows were dropped due to invalid dates after parsing.")
        empty = pd.DataFrame(columns=rows + periods)
        return {
            "meta": meta,
            "table": empty,
        }

    # Skapa periodkolumner
    df["period_p"] = df[date_col].dt.to_period("M")
    df["period"] = df["period_p"].astype(str)
    df["year"] = df[date_col].dt.year

    # Fyll null i dimensioner
    df = _normalize_dim_nulls(df, rows)

    # Först: aggregera till månad per rows (+ year)
    monthly = (
        df.groupby(rows + ["year", "period_p"], dropna=False, as_index=False)["amount"]
        .sum()
    )
    monthly["period"] = monthly["period_p"].astype(str)

    # YTD: densifiera månader (så saknade månader carry-forwardar istället för att bli 0)
    if compare_mode == "ytd":
        req_pi = pd.PeriodIndex(periods, freq="M")

        # Bygg "kalender" per år: Jan -> max efterfrågad månad i det året
        months_parts = []
        for y in sorted(pd.unique(req_pi.year)):
            max_p = req_pi[req_pi.year == y].max()
            months_y = pd.period_range(pd.Period(f"{y}-01", freq="M"), max_p, freq="M")
            months_parts.append(pd.DataFrame({"year": y, "period_p": months_y}))
        all_months = pd.concat(months_parts, ignore_index=True)

        key_cols = rows + ["year"]

        # Alla grupper per år (rows + year)
        groups = monthly[key_cols].drop_duplicates()

        # Densifiera INOM samma year
        full = groups.merge(all_months, on="year", how="inner")

        # Left-join in belopp; saknade månader => 0
        monthly = full.merge(
            monthly[key_cols + ["period_p", "amount"]],
            on=key_cols + ["period_p"],
            how="left",
        )
        monthly["amount"] = monthly["amount"].fillna(0)

        # Cumsum inom (rows + year)
        monthly = monthly.sort_values(key_cols + ["period_p"])
        monthly["amount"] = monthly.groupby(key_cols, dropna=False)["amount"].cumsum()
        monthly["period"] = monthly["period_p"].astype(str)

    # Behåll bara requested periods
    keep = set(periods)
    slim = monthly[monthly["period"].isin(keep)].copy()

    # Pivot till wide-format
    wide = (
        slim.pivot_table(
            index=rows,
            columns="period",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    # Säkerställ att alla perioder finns som kolumner och i rätt ordning
    for p in periods:
        if p not in wide.columns:
            wide[p] = 0
    wide = wide[rows + periods]

    # Totals
    if include_total:
        base_wide = wide.copy()
        total_rows = []

        # Subtotal per konto_typ om det är med i rows och det finns fler dimensioner
        if "konto_typ" in rows and len(rows) > 1:
            period_cols = periods
            subtotal = base_wide.groupby("konto_typ", dropna=False)[period_cols].sum().reset_index()
            for r in rows:
                if r != "konto_typ":
                    subtotal[r] = "__TOTAL__"
            subtotal = subtotal[rows + period_cols]
            total_rows.append(subtotal)

        # Grand total
        gt = base_wide[periods].sum().to_frame().T
        gt_row = {}
        gt_row[rows[0]] = "__TOTAL__"
        for r in rows[1:]:
            gt_row[r] = ""
        for k, v in gt_row.items():
            gt[k] = v
        gt = gt[rows + periods]
        total_rows.append(gt)

        wide = pd.concat([wide] + total_rows, ignore_index=True)

    return {
        "meta": meta,
        "table": wide,
    }