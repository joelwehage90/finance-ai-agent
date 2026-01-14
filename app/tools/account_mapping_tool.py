from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import re

from ..supabase_client import supabase


def account_mapping_query(
    *,
    mode: str = "hierarchy",
    accounts: Optional[List[int]] = None,
    rr_level_1: Optional[str] = None,
    rr_level_2: Optional[str] = None,
    konto_typ: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """
    mode="hierarchy": returnerar rr_level_1/rr_level_2/konto_typ (+ antal konton)
    mode="lookup": returnerar konton + mappingfält
    """
    mode = (mode or "hierarchy").strip().lower()
    if mode not in {"hierarchy", "lookup"}:
        raise ValueError("mode måste vara 'hierarchy' eller 'lookup'.")

    q = supabase.table("account_mapping").select("konto,kontonamn,rr_level_1,rr_level_2,konto_typ")

    # filters (primärt för lookup, men kan även användas för hierarchy om du vill)
    if accounts:
        q = q.in_("konto", accounts)
    if rr_level_1:
        q = q.eq("rr_level_1", rr_level_1)
    if rr_level_2:
        q = q.eq("rr_level_2", rr_level_2)
    if konto_typ:
        q = q.eq("konto_typ", konto_typ)

    # NOTE: Our PostgREST client wrapper does not reliably support OR/ILIKE helpers across versions.
    # We therefore do a best-effort local search (table is small) to avoid 500s.
    # When `search` is provided, we fetch up to 2000 rows and filter in Python.
    q = q.limit(2000 if (search and str(search).strip()) else min(limit, 2000))
    resp = q.execute()
    rows = resp.data or []

    # Local, case-insensitive search across common columns (best-effort).
    if search and str(search).strip():
        terms = [t for t in re.split(r"[\s,]+", str(search).strip()) if t]
        terms_cf = [t.casefold() for t in terms if t]
        cols_to_search = ["kontonamn", "rr_level_1", "rr_level_2", "konto_typ"]

        def _matches(r: Dict[str, Any]) -> bool:
            hay = [(str(r.get(c) or "")).casefold() for c in cols_to_search]
            # All terms must match somewhere (AND across terms, OR across columns).
            for tcf in terms_cf:
                if not any(tcf in h for h in hay):
                    return False
            return True

        rows = [r for r in rows if isinstance(r, dict) and _matches(r)]

    if mode == "lookup":
        # Make tool output compatible with /tools/format by providing a table+columns alias.
        # Keep `items` for backward compatibility.
        cols = ["konto", "kontonamn", "rr_level_1", "rr_level_2", "konto_typ"]
        meta = {
            "mode": mode,
            "returned": len(rows),
            "limit": limit,
            # Formatting hints: treat all columns as dims (avoid unit scaling on ints like konto).
            "rows": cols,
            "value_columns": [],
        }
        return {
            "meta": meta,
            "columns": cols,
            "table": rows,
            "items": rows,
        }

    # hierarchy: groupa och räkna antal konton
    df = pd.DataFrame(rows)
    if df.empty:
        cols = ["rr_level_1", "rr_level_2", "konto_typ", "account_count"]
        meta = {"mode": mode, "returned": 0, "limit": limit, "rows": cols, "value_columns": []}
        return {"meta": meta, "columns": cols, "table": [], "items": []}

    g = (
        df.groupby(["rr_level_1", "rr_level_2", "konto_typ"], dropna=False)["konto"]
        .nunique()
        .reset_index()
        .rename(columns={"konto": "account_count"})
        .sort_values(["rr_level_1", "rr_level_2", "konto_typ"])
    )

    items = g.to_dict(orient="records")
    out_items = items[:limit]
    cols = ["rr_level_1", "rr_level_2", "konto_typ", "account_count"]
    meta = {
        "mode": mode,
        "returned": len(out_items),
        "limit": limit,
        # Formatting hints: treat all columns as dims (account_count should not be unit-scaled).
        "rows": cols,
        "value_columns": [],
    }
    return {"meta": meta, "columns": cols, "table": out_items, "items": out_items}