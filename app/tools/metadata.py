from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from ..supabase_client import supabase


def _rpc_single(fn_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliserar supabase.rpc(...) till en dict.
    RPC kan ibland returnera dict eller list[dict] beroende på funktion/klientversion.
    """
    resp = supabase.rpc(fn_name, args).execute()
    data = resp.data

    if data is None:
        return {}

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        if len(data) == 0:
            return {}
        if isinstance(data[0], dict):
            return data[0]

    return {"raw": data}


@lru_cache(maxsize=64)
def get_relation_docs(schema: str, relname: str) -> Dict[str, Any]:
    """
    Hämtar COMMENT ON för vy/tabell + kolumnkommentarer via SQL-funktionen ai_get_relation_docs.
    Cacheas per process.
    """
    return _rpc_single(
        "ai_get_relation_docs",
        {"p_schema": schema, "p_relname": relname},
    )


def pick_column_docs(all_docs: Dict[str, Any], used_columns: list[str]) -> Dict[str, Optional[str]]:
    """
    Returnerar endast docs för kolumner som faktiskt används i output.
    """
    cols = all_docs.get("columns") or {}
    return {c: cols.get(c) for c in used_columns}


def attach_meta(
    result: Dict[str, Any],
    *,
    schema: str,
    relation: str,
    rows: list[dict],
    extra_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Lägger till result['meta'] med källa + docs + semantiska hints.
    """
    used_cols = list(rows[0].keys()) if rows else []
    docs = get_relation_docs(schema, relation)

    meta = result.setdefault("meta", {})
    meta["source"] = {"schema": schema, "relation": relation}
    meta["docs"] = {
        "relation_comment": docs.get("relation_comment"),
        "columns": pick_column_docs(docs, used_cols),
    }

    # Alltid bra att vara explicit i finance
    hints = meta.setdefault("semantic_hints", {})
    hints.setdefault("amount_sign", "revenue_positive_cost_negative")

    if extra_hints:
        hints.update(extra_hints)

    meta.setdefault("warnings", [])
    return result