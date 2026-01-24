from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..supabase_client import supabase
from .format_spec import FormatSpec, default_format_spec
from .interpret_format_request_openai import interpret_format_request_openai, apply_partial_format_overrides


_TOOL_DEFAULT_FORMAT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # P&L: amounts; default should be readable for humans (mkr) and preserve row order.
    "income_statement_tool": {"unit": "msek", "decimals": 1, "include_totals": True, "sort": None, "top_n": None},
    # Variance: multi tables; default should not override tool-provided sorting hints.
    "variance_tool": {"unit": "msek", "decimals": 1, "include_totals": False, "sort": None, "top_n": None},
    # Lookup-style tables: never scale numbers, never add totals, keep as-is.
    "definitions_tool": {"include_totals": False, "sort": None, "top_n": None},
    "account_mapping_tool": {"include_totals": False, "sort": None, "top_n": None},
    # NL→SQL: preserve database ordering and avoid totals semantics.
    "nl_sql_tool": {"unit": "sek", "decimals": 0, "include_totals": None, "sort": None, "top_n": None},
}


def _res_data(res: Any) -> List[Dict[str, Any]]:
    """
    Normalize SupabaseResponse -> list[dict].
    """
    try:
        data = getattr(res, "data", None)
    except Exception:
        data = None
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _fetch_tool_name(source_tool_run_id: str) -> Optional[str]:
    """
    Best-effort lookup of tool_runs.tool_name for a given tool_run id.
    Used to apply sensible per-tool defaults when no prior presentation artifact exists.
    """
    try:
        res = (
            supabase.table("tool_runs")
            .select("tool_name")
            .eq("id", str(source_tool_run_id))
            .limit(1)
            .execute()
        )
        rows = _res_data(res)
        if rows:
            tn = rows[0].get("tool_name")
            if tn is None:
                return None
            s = str(tn).strip()
            return s or None
    except Exception:
        return None
    return None


def _apply_tool_defaults(
    *,
    base_spec: FormatSpec,
    source_tool_run_id: str,
) -> FormatSpec:
    """
    Apply per-tool default overrides on top of default_format_spec().

    Important semantics:
    - This is applied BEFORE interpreting format_request, so user/LLM changes can override the defaults.
    - If there is an existing presentation artifact for the turn, we do NOT apply tool defaults here
      (we want the incremental base to be the latest artifact spec).
    """
    tool_name = _fetch_tool_name(source_tool_run_id)
    if not tool_name:
        return base_spec
    raw = _TOOL_DEFAULT_FORMAT_OVERRIDES.get(tool_name)
    if not isinstance(raw, dict) or not raw:
        return base_spec
    try:
        applied = apply_partial_format_overrides(base=base_spec, raw=raw)
        return applied["spec"]
    except Exception:
        return base_spec


def _merge_nested_overrides(base: FormatSpec, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge semantics for nested fields when doing incremental formatting:
    - rename_columns: dict-merge (new keys override old keys)
    - derive: merge by derived column name (new entries override same-name old entries)

    Note: if raw explicitly sets these fields to None, we honor that (clears).
    """
    out = dict(raw)

    # rename_columns: dict merge
    if "rename_columns" in out:
        rc = out.get("rename_columns")
        if rc is None:
            # explicit clear
            pass
        elif isinstance(rc, dict):
            base_rc = base.rename_columns or {}
            merged_rc = dict(base_rc)
            for k, v in rc.items():
                k2 = str(k).strip()
                v2 = str(v).strip()
                if not k2 or not v2:
                    continue
                merged_rc[k2] = v2
            out["rename_columns"] = merged_rc or None

    # column_decimals: dict merge
    if "column_decimals" in out:
        cd = out.get("column_decimals")
        if cd is None:
            # explicit clear
            pass
        elif isinstance(cd, dict):
            base_cd = base.column_decimals or {}
            merged_cd = dict(base_cd)
            for k, v in cd.items():
                k2 = str(k).strip()
                if not k2:
                    continue
                try:
                    d = int(v)
                except Exception:
                    continue
                if 0 <= d <= 3:
                    merged_cd[k2] = d
            out["column_decimals"] = merged_cd or None

    # derive: merge-by-name
    if "derive" in out:
        dv = out.get("derive")
        if dv is None:
            # explicit clear
            pass
        elif isinstance(dv, list):
            merged_by_name: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []

            # start from base
            for d in base.derive or []:
                d0 = d.model_dump(mode="json")
                name = str(d0.get("name") or "").strip()
                if not name:
                    continue
                if name not in merged_by_name:
                    order.append(name)
                merged_by_name[name] = d0

            # apply incoming
            for item in dv:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                if name not in merged_by_name:
                    order.append(name)
                merged_by_name[name] = dict(item)

            out["derive"] = [merged_by_name[n] for n in order if n in merged_by_name] or None

    # filters: merge-by-id (if present) else by signature (col|op|value)
    if "filters" in out:
        fv = out.get("filters")
        if fv is None:
            # explicit clear
            pass
        elif isinstance(fv, list):
            def _key(f: Dict[str, Any]) -> str:
                fid = str(f.get("id") or "").strip()
                if fid:
                    return f"id:{fid}"
                col = str(f.get("col") or "").strip()
                op = str(f.get("op") or "").strip()
                val = f.get("value")
                # stable-ish value repr
                if isinstance(val, list):
                    v2 = ",".join([str(x) for x in val])
                else:
                    v2 = str(val)
                return f"sig:{col}|{op}|{v2}"

            merged: Dict[str, Dict[str, Any]] = {}
            order2: List[str] = []

            for f in base.filters or []:
                d0 = f.model_dump(mode="json")
                k0 = _key(d0)
                if k0 not in merged:
                    order2.append(k0)
                merged[k0] = d0

            for item in fv:
                if not isinstance(item, dict):
                    continue
                k1 = _key(item)
                if k1 not in merged:
                    order2.append(k1)
                merged[k1] = dict(item)

            out["filters"] = [merged[k] for k in order2 if k in merged] or None

    # filter_groups: merge-by-id (if present) else by signature (op + rules signature)
    if "filter_groups" in out:
        gv = out.get("filter_groups")
        if gv is None:
            pass
        elif isinstance(gv, list):
            def _grp_key(g: Dict[str, Any]) -> str:
                gid = str(g.get("id") or "").strip()
                if gid:
                    return f"id:{gid}"
                op = str(g.get("op") or "or").strip().lower()
                rules = g.get("rules") if isinstance(g.get("rules"), list) else []
                parts: List[str] = []
                for r in rules:
                    if not isinstance(r, dict):
                        continue
                    col = str(r.get("col") or "").strip()
                    rop = str(r.get("op") or "").strip()
                    val = r.get("value")
                    if isinstance(val, list):
                        v2 = ",".join([str(x) for x in val])
                    else:
                        v2 = str(val)
                    parts.append(f"{col}|{rop}|{v2}")
                return f"sig:{op}:" + ";".join(parts)

            merged_g: Dict[str, Dict[str, Any]] = {}
            order_g: List[str] = []
            for g in base.filter_groups or []:
                d0 = g.model_dump(mode="json")
                k0 = _grp_key(d0)
                if k0 not in merged_g:
                    order_g.append(k0)
                merged_g[k0] = d0

            for item in gv:
                if not isinstance(item, dict):
                    continue
                k1 = _grp_key(item)
                if k1 not in merged_g:
                    order_g.append(k1)
                merged_g[k1] = dict(item)

            out["filter_groups"] = [merged_g[k] for k in order_g if k in merged_g] or None

    # filter_expr: replace semantics (if provided), explicit None clears.
    if "filter_expr" in out:
        # keep as-is; validation happens in FormatSpec
        pass

    return out


def _fetch_latest_presentation_artifact(
    session_id: str,
    turn_id: int,
    *,
    source_tool_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recently updated presentation_table artifact for (session_id, turn_id, source_tool_name).
    Note: we don't have an is_active flag; singleton is enforced via upsert + unique index in DB.
    """
    q = (
        supabase.table("artifacts")
        .select("id,updated_at,source_tool_run_id,format_spec,source_tool_name")
        .eq("session_id", session_id)
        .eq("turn_id", int(turn_id))
        .eq("artifact_type", "presentation_table")
        .order("updated_at", desc=True)
        .limit(1)
    )
    # Backwards-safe fallback: if tool_name is unknown, we return the latest across tools (legacy behavior).
    if source_tool_name:
        q = q.eq("source_tool_name", str(source_tool_name))
    res = q.execute()
    rows = _res_data(res)
    if rows:
        return rows[0]
    return None


def _coerce_format_spec(obj: Any) -> Optional[FormatSpec]:
    if not obj or not isinstance(obj, dict):
        return None
    try:
        # Stored format_spec is expected to be complete, but we still merge onto defaults for safety.
        base = default_format_spec().model_dump(mode="json")
        base.update(obj)
        return FormatSpec.model_validate(base)
    except Exception:
        return None


def resolve_incremental_format_spec(
    *,
    session_id: str,
    turn_id: int,
    source_tool_run_id: str,
    format_request: Optional[str],
    format_spec_overrides: Optional[Dict[str, Any]],
    reset: bool,
    apply_on_latest_presentation_data: bool = True,
) -> Dict[str, Any]:
    """
    Build an *incremental* FormatSpec for reformatting.

    Rules:
    - Base spec is latest presentation_table.format_spec for this turn (if exists), otherwise defaults.
    - If reset=True (or LLM triggers reset), base spec becomes defaults.
    - LLM returns *partial* changes; we merge them onto base spec.
    - Explicit format_spec_overrides are applied last (also as partial) onto the current spec.
    - If apply_on_latest_presentation_data=True, effective source_tool_run_id defaults to the latest
      presentation_table's source_tool_run_id (to match "apply on latest presentation table").
      If False, we always use the provided source_tool_run_id (useful for auto-formatting right after a tool run).
    """
    notes: List[str] = []
    scope_tool_name = _fetch_tool_name(str(source_tool_run_id))
    if not scope_tool_name:
        notes.append("Could not determine tool_name for source_tool_run_id; using latest presentation artifact across tools.")
    base_art = _fetch_latest_presentation_artifact(
        session_id=session_id,
        turn_id=turn_id,
        source_tool_name=scope_tool_name,
    )
    base_spec = default_format_spec()

    effective_source_tool_run_id = source_tool_run_id
    parent_artifact_id: Optional[str] = None

    if base_art:
        parent_artifact_id = base_art.get("id")
        stored_spec = _coerce_format_spec(base_art.get("format_spec"))
        if stored_spec is not None:
            base_spec = stored_spec
        if apply_on_latest_presentation_data and base_art.get("source_tool_run_id"):
            # Only default to the latest presentation artifact's source when the caller
            # is not explicitly pointing at a different tool_run.
            #
            # This preserves the UI behavior: if you choose a specific tool_run to format,
            # we should *not* silently switch to some previous artifact's source.
            base_src = str(base_art.get("source_tool_run_id"))
            if str(source_tool_run_id) == base_src:
                effective_source_tool_run_id = base_src

    created_mode = "manual"
    if reset:
        base_spec = default_format_spec()
        base_spec = _apply_tool_defaults(base_spec=base_spec, source_tool_run_id=effective_source_tool_run_id)

    # If there is no base artifact (or the stored spec was missing/invalid), we're starting from defaults.
    # Apply per-tool defaults BEFORE any LLM interpretation so that "top 5" still keeps unit=mkr for P&L, etc.
    if not base_art:
        base_spec = _apply_tool_defaults(base_spec=base_spec, source_tool_run_id=effective_source_tool_run_id)

    # LLM: partial changes (and possible reset)
    llm_reset = False
    llm_partial: Optional[Dict[str, Any]] = None
    llm_notes: List[str] = []
    validation_notes: List[str] = []
    if format_request and str(format_request).strip():
        created_mode = "interpret_request"
        interp = interpret_format_request_openai(
            source_tool_run_id=effective_source_tool_run_id,
            format_request=str(format_request),
            base_spec=base_spec.model_dump(mode="json"),
        )
        llm_reset = bool(interp.get("reset"))
        llm_partial = interp.get("partial") if isinstance(interp.get("partial"), dict) else None
        n2 = interp.get("notes") or []
        llm_notes = [str(x) for x in n2 if isinstance(x, str) and str(x).strip()]
        notes.extend(llm_notes)

        # If LLM asked to reset, restart from defaults BEFORE applying partial.
        if llm_reset:
            base_spec = default_format_spec()
            base_spec = _apply_tool_defaults(base_spec=base_spec, source_tool_run_id=effective_source_tool_run_id)
        partial = interp.get("partial") or {}
        if isinstance(partial, dict):
            partial = _merge_nested_overrides(base_spec, partial)
        applied = apply_partial_format_overrides(base=base_spec, raw=partial if isinstance(partial, dict) else {})
        base_spec = applied["spec"]
        validation_notes = [str(x) for x in (applied.get("notes") or []) if isinstance(x, str) and str(x).strip()]
        notes.extend(validation_notes)

    # Manual/explicit overrides applied last (also incremental)
    if format_spec_overrides is not None:
        raw2 = format_spec_overrides
        if isinstance(raw2, dict):
            raw2 = _merge_nested_overrides(base_spec, raw2)
        applied2 = apply_partial_format_overrides(base=base_spec, raw=raw2)
        base_spec = applied2["spec"]
        notes.extend([str(x) for x in (applied2.get("notes") or []) if isinstance(x, str) and str(x).strip()])

    # De-dupe notes (stable)
    deduped: List[str] = []
    seen = set()
    for n in notes:
        n2 = str(n).strip()
        if not n2 or n2 in seen:
            continue
        seen.add(n2)
        deduped.append(n2)

    interpretation = None
    if format_request and str(format_request).strip():
        interpretation = {
            "request": str(format_request),
            "partial": llm_partial,
            "reset": bool(llm_reset),
            "llm_notes": llm_notes,
            "validation_notes": validation_notes,
        }

    return {
        "spec": base_spec,
        "notes": deduped,
        "created_mode": created_mode,
        "effective_source_tool_run_id": effective_source_tool_run_id,
        "parent_artifact_id": parent_artifact_id,
        "reset": bool(reset or llm_reset),
        "interpretation": interpretation,
    }


