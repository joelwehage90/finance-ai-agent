from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..supabase_client import supabase


def _clean_terms(terms: Optional[Sequence[str]]) -> List[str]:
    if not terms:
        return []
    out = []
    for t in terms:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            out.append(s)
    # unik + stabil ordning
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _clean_search(search: Optional[str]) -> Optional[str]:
    if not search:
        return None
    s = str(search).strip()
    if not s:
        return None
    # PostgREST or_ använder kommatecken som separator -> sanera
    s = s.replace(",", " ")
    # Hårdnad sanering: normalisera till "vanlig text"
    # - ersätt specialtecken med mellanslag
    # - allowlist: bokstäver/siffror/_/-/mellanslag + ÅÄÖåäö
    # - kollapsa whitespace
    s = re.sub(r"[()=*<>|/\\\[\]{}:;\"'`~!@#$%^&+?.]", " ", s)
    s = re.sub(r"[^0-9A-Za-z_\- ÅÄÖåäö]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _postgrest_ilike_pattern(search: str) -> str:
    # PostgREST ILIKE wildcard är * (inte %)
    return f"*{search}*"


def _to_text(v: Any) -> Optional[str]:
    """
    Best-effort coercion into a human-readable string.
    Returns None for empty/None values.
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, list):
        parts: List[str] = []
        for x in v:
            xs = _to_text(x)
            if xs:
                parts.append(xs)
        s = "\n".join(parts).strip()
        return s if s else None
    if isinstance(v, dict):
        try:
            s = json.dumps(v, ensure_ascii=False)
            s = s.strip()
            return s if s else None
        except Exception:
            return str(v)
    s = str(v).strip()
    return s if s else None


def _tokenize_search(search: str) -> List[str]:
    # Split on whitespace. Search is already sanitized in _clean_search.
    toks = [t.strip() for t in str(search).split() if t and t.strip()]
    # unique + stable (casefold)
    seen = set()
    out: List[str] = []
    for t in toks:
        t2 = t.casefold()
        if t2 not in seen:
            out.append(t)
            seen.add(t2)
    return out[:12]

def _local_token_match_filter(
    rows: List[Dict[str, Any]],
    *,
    search: str,
    columns: List[str],
) -> List[Dict[str, Any]]:
    """
    Local, case-insensitive token search:
    - OR across columns
    - require at least min(2, len(tokens)) matched tokens
    - rows are returned in best-effort relevance order:
        - higher matched-token count first
        - then stable original order
    """
    toks = _tokenize_search(search)
    if not toks:
        return rows
    toks_cf = [t.casefold() for t in toks]
    required = min(2, len(toks_cf))

    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        hay = [(str(r.get(c) or "")).casefold() for c in columns]
        matched = 0
        for tcf in toks_cf:
            if any(tcf in h for h in hay):
                matched += 1
        if matched >= required:
            scored.append((matched, idx, r))

    # Sort by score desc, then stable original order
    scored.sort(key=lambda x: (-int(x[0]), int(x[1])))
    return [r for _, _, r in scored]


def _dedupe_by_key(rows: List[Dict[str, Any]], *, key_field: str) -> List[Dict[str, Any]]:
    """
    Dedupe a list of rows by a case-insensitive key field, keeping first occurrence.
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = str(r.get(key_field) or "").strip()
        if not k:
            continue
        k_cf = k.casefold()
        if k_cf in seen:
            continue
        seen.add(k_cf)
        out.append(r)
    return out


def _fetch_token_candidates(
    *,
    table: str,
    select: str,
    token: str,
    columns: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Fetch candidates for a single token using server-side OR + ILIKE across the given columns.
    Token is assumed to be sanitized.
    """
    pat = _postgrest_ilike_pattern(token)
    ors = ",".join([f"{c}.ilike.{pat}" for c in columns])
    resp = supabase.table(table).select(select).or_(ors).limit(int(limit)).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Supabase error ({table} token search): {resp.error}")
    return resp.data or []


@lru_cache(maxsize=128)
def _fetch_business_exact(terms_key: Tuple[str, ...]) -> List[Dict[str, Any]]:
    if not terms_key:
        return []
    resp = (
        supabase.table("business_definitions")
        .select("term,definition,examples,example_sql,owner,updated_at")
        .in_("term", list(terms_key))
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(f"Supabase error (business_definitions exact): {resp.error}")
    return resp.data or []


@lru_cache(maxsize=128)
def _fetch_metrics_exact(terms_key: Tuple[str, ...]) -> List[Dict[str, Any]]:
    if not terms_key:
        return []
    resp = (
        supabase.table("metric_definitions")
        .select("metric_name,definition,formula_text,scope_filters,caveats,example_questions,updated_at")
        .in_("metric_name", list(terms_key))
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(f"Supabase error (metric_definitions exact): {resp.error}")
    return resp.data or []


def _fetch_business_search(search: str, limit: int) -> List[Dict[str, Any]]:
    pat = _postgrest_ilike_pattern(search)
    # term/definition/examples/owner (NOTE: do NOT search in example_sql)
    resp = (
        supabase.table("business_definitions")
        .select("term,definition,examples,example_sql,owner,updated_at")
        .or_(f"term.ilike.{pat},definition.ilike.{pat},examples.ilike.{pat},owner.ilike.{pat}")
        .limit(limit)
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(f"Supabase error (business_definitions search): {resp.error}")
    return resp.data or []


def _fetch_metrics_search(search: str, limit: int) -> List[Dict[str, Any]]:
    pat = _postgrest_ilike_pattern(search)
    resp = (
        supabase.table("metric_definitions")
        .select("metric_name,definition,formula_text,scope_filters,caveats,example_questions,updated_at")
        .or_(
            f"metric_name.ilike.{pat},definition.ilike.{pat},formula_text.ilike.{pat},"
            f"scope_filters.ilike.{pat},caveats.ilike.{pat},example_questions.ilike.{pat}"
        )
        .limit(limit)
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(f"Supabase error (metric_definitions search): {resp.error}")
    return resp.data or []


def definitions_lookup(
    *,
    terms: Optional[Sequence[str]] = None,
    search: Optional[str] = None,
    include_business: bool = True,
    include_metrics: bool = True,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Hämtar definitioner från:
      - business_definitions (term)
      - metric_definitions (metric_name)

    Strategi:
      1) exact match på terms (prioriteras)
      2) fritext-sökning (ILIKE) på search (kompletterar)
      3) dedupe + limit

    Return:
      {"meta": {...}, "items": [ ... ]}
    """
    if limit <= 0:
        raise ValueError("limit måste vara > 0")

    t = _clean_terms(terms)
    s = _clean_search(search)

    if not t and not s:
        raise ValueError("Du måste ange minst en term (terms) eller en söksträng (search).")

    warnings: List[str] = []
    items: List[Dict[str, Any]] = []

    # 1) Exact match (cachead)
    # Canonicalize for cache hit-rate
    terms_key = tuple(sorted(t, key=lambda x: str(x).casefold()))
    try:
        if include_business and t:
            for row in _fetch_business_exact(terms_key):
                name = _to_text(row.get("term"))
                definition = _to_text(row.get("definition"))
                if not name or not definition:
                    continue
                items.append({
                    "kind": "business",
                    "name": name,
                    "definition": definition,
                    "examples": _to_text(row.get("examples")),
                    "example_sql": _to_text(row.get("example_sql")),
                    "owner": _to_text(row.get("owner")),
                    "updated_at": _to_text(row.get("updated_at")),
                    "_match": "exact",
                })
        if include_metrics and t:
            for row in _fetch_metrics_exact(terms_key):
                name = _to_text(row.get("metric_name"))
                definition = _to_text(row.get("definition"))
                if not name or not definition:
                    continue
                items.append({
                    "kind": "metric",
                    "name": name,
                    "definition": definition,
                    "formula_text": _to_text(row.get("formula_text")),
                    "scope_filters": _to_text(row.get("scope_filters")),
                    "caveats": _to_text(row.get("caveats")),
                    "example_questions": _to_text(row.get("example_questions")),
                    "updated_at": _to_text(row.get("updated_at")),
                    "_match": "exact",
                })
    except Exception as e:
        warnings.append(f"Exact lookup failed: {type(e).__name__}: {e}")

    # 2) Search (ej cachead, men limitad)
    # Vi tar lite “headroom” så vi kan dedupe och ändå fylla limit
    search_limit = max(10, min(50, limit * 2))
    if s:
        # NOTE:
        # - Search is token-based (broad) and requires at least min(2, len(tokens)) matched tokens per row.
        # - We try server-side OR/ILIKE per token to fetch candidates, and ALWAYS apply local token scoring/filtering.
        # - If server-side fails, we fall back to bounded fetch + local scoring/filtering.
        try:
            toks = _tokenize_search(s)
            if len(toks) > 12:
                toks = toks[:12]
                warnings.append("Search had many tokens; using first 12.")

            # Collect candidates per token and dedupe by key.
            if include_business:
                cand: List[Dict[str, Any]] = []
                for tok in toks:
                    cand.extend(
                        _fetch_token_candidates(
                            table="business_definitions",
                            select="term,definition,examples,example_sql,owner,updated_at",
                            token=tok,
                            columns=["term", "definition", "examples", "owner"],  # IMPORTANT: no example_sql
                            limit=max(10, int(search_limit)),
                        )
                    )
                cand = _dedupe_by_key(cand, key_field="term")
                # Always apply local token scoring/filtering
                cand = _local_token_match_filter(
                    cand,
                    search=s,
                    columns=["term", "definition", "examples", "owner"],  # IMPORTANT: no example_sql
                )
                for row in cand[:search_limit]:
                    name = _to_text(row.get("term"))
                    definition = _to_text(row.get("definition"))
                    if not name or not definition:
                        continue
                    items.append(
                        {
                            "kind": "business",
                            "name": name,
                            "definition": definition,
                            "examples": _to_text(row.get("examples")),
                            "example_sql": _to_text(row.get("example_sql")),  # kept for compatibility; NOT searched
                            "owner": _to_text(row.get("owner")),
                            "updated_at": _to_text(row.get("updated_at")),
                            "_match": "search",
                        }
                    )

            if include_metrics:
                cand2: List[Dict[str, Any]] = []
                for tok in toks:
                    cand2.extend(
                        _fetch_token_candidates(
                            table="metric_definitions",
                            select="metric_name,definition,formula_text,scope_filters,caveats,example_questions,updated_at",
                            token=tok,
                            columns=[
                                "metric_name",
                                "definition",
                                "formula_text",
                                "scope_filters",
                                "caveats",
                                "example_questions",
                            ],
                            limit=max(10, int(search_limit)),
                        )
                    )
                cand2 = _dedupe_by_key(cand2, key_field="metric_name")
                cand2 = _local_token_match_filter(
                    cand2,
                    search=s,
                    columns=["metric_name", "definition", "formula_text", "scope_filters", "caveats", "example_questions"],
                )
                for row in cand2[:search_limit]:
                    name = _to_text(row.get("metric_name"))
                    definition = _to_text(row.get("definition"))
                    if not name or not definition:
                        continue
                    items.append(
                        {
                            "kind": "metric",
                            "name": name,
                            "definition": definition,
                            "formula_text": _to_text(row.get("formula_text")),
                            "scope_filters": _to_text(row.get("scope_filters")),
                            "caveats": _to_text(row.get("caveats")),
                            "example_questions": _to_text(row.get("example_questions")),
                            "updated_at": _to_text(row.get("updated_at")),
                            "_match": "search",
                        }
                    )
        except Exception as e:
            warnings.append(f"Search lookup failed (server-side OR/ILIKE): {type(e).__name__}: {e}")
            # Fallback: bounded fetch + local scoring/filtering (best-effort; avoids 500s)
            try:
                fetch_limit = min(2000, max(200, search_limit * 20))
                if include_business:
                    r = (
                        supabase.table("business_definitions")
                        .select("term,definition,examples,example_sql,owner,updated_at")
                        .limit(fetch_limit)
                        .execute()
                    )
                    if getattr(r, "error", None):
                        warnings.append(f"Supabase error (business_definitions fallback): {r.error}")
                    rows = r.data or []
                    rows = _local_token_match_filter(rows, search=s, columns=["term", "definition", "examples", "owner"])
                    for row in rows[:search_limit]:
                        name = _to_text(row.get("term"))
                        definition = _to_text(row.get("definition"))
                        if not name or not definition:
                            continue
                        items.append({
                            "kind": "business",
                            "name": name,
                            "definition": definition,
                            "examples": _to_text(row.get("examples")),
                            "example_sql": _to_text(row.get("example_sql")),
                            "owner": _to_text(row.get("owner")),
                            "updated_at": _to_text(row.get("updated_at")),
                            "_match": "search",
                        })
                if include_metrics:
                    r = (
                        supabase.table("metric_definitions")
                        .select("metric_name,definition,formula_text,scope_filters,caveats,example_questions,updated_at")
                        .limit(fetch_limit)
                        .execute()
                    )
                    if getattr(r, "error", None):
                        warnings.append(f"Supabase error (metric_definitions fallback): {r.error}")
                    rows = r.data or []
                    rows = _local_token_match_filter(
                        rows,
                        search=s,
                        columns=["metric_name", "definition", "formula_text", "scope_filters", "caveats", "example_questions"],
                    )
                    for row in rows[:search_limit]:
                        name = _to_text(row.get("metric_name"))
                        definition = _to_text(row.get("definition"))
                        if not name or not definition:
                            continue
                        items.append({
                            "kind": "metric",
                            "name": name,
                            "definition": definition,
                            "formula_text": _to_text(row.get("formula_text")),
                            "scope_filters": _to_text(row.get("scope_filters")),
                            "caveats": _to_text(row.get("caveats")),
                            "example_questions": _to_text(row.get("example_questions")),
                            "updated_at": _to_text(row.get("updated_at")),
                            "_match": "search",
                        })
            except Exception as e2:
                warnings.append(f"Search lookup failed (local fallback): {type(e2).__name__}: {e2}")

    # 3) Dedupe + sort: exact först, sedan search
    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for it in items:
        # Dedupe should be case-insensitive on name to avoid duplicates from mixed casing.
        key = (it.get("kind") or "", str(it.get("name") or "").casefold())
        if key == ("", ""):
            continue
        # exact ska vinna över search
        if key not in dedup:
            dedup[key] = it
        else:
            if dedup[key].get("_match") != "exact" and it.get("_match") == "exact":
                dedup[key] = it

    out_items = list(dedup.values())
    out_items.sort(key=lambda x: 0 if x.get("_match") == "exact" else 1)

    out_items = out_items[:limit]

    # Make tool output compatible with /tools/format by providing a table+columns alias.
    # Keep `items` for backward compatibility.
    # IMPORTANT: These columns define the formatter-friendly table shape.
    # - Do NOT include example_sql here (per requirement). It may still exist in `items` for compatibility.
    cols = [
        "kind",
        "name",
        "definition",
        "examples",
        "owner",
        "formula_text",
        "scope_filters",
        "caveats",
        "example_questions",
        "updated_at",
        "_match",
    ]
    meta = {
        "terms": t,
        "search": s,
        "search_tokens": _tokenize_search(s) if s else [],
        "required_token_matches": (min(2, len(_tokenize_search(s))) if s else 0),
        "include_business": include_business,
        "include_metrics": include_metrics,
        "limit": limit,
        "returned": len(out_items),
        "warnings": warnings,
        # Formatting hints: treat everything as dims (avoid unit scaling; values are mostly strings).
        "rows": cols,
        "value_columns": [],
    }
    table_rows = [{c: it.get(c) for c in cols} for it in out_items]
    return {"meta": meta, "columns": cols, "table": table_rows, "items": out_items}