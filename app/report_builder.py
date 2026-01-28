from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .supabase_client import supabase
from .tools.income_statement_tool import pnl_tables
from .tools.variance_tool import variance_tables
from .schemas.pnl import PnlRequest
from .schemas.variance import VarianceRequest
from .tool_logging import log_tool_call

logger = logging.getLogger(__name__)


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


def _df_to_records(df: Any) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    if df is None:
        return []
    if pd is not None and isinstance(df, pd.DataFrame):
        if df.empty:
            return []
        df2 = df.where(pd.notnull(df), None)
        return df2.to_dict(orient="records")
    return []


def _insert_report_run(
    *,
    period: str,
    report_spec: Dict[str, Any],
    customer_id: Optional[str] = None,
) -> str:
    year_s, month_s = period.split("-")
    payload = {
        "period_year": int(year_s),
        "period_month": int(month_s),
        "status": "draft",
        "report_spec": report_spec,
        "builder_version": "report_builder_v0",
    }
    if customer_id is not None:
        payload["customer_id"] = customer_id
    res = supabase.table("report_runs").insert(payload).execute()
    err = _res_error(res)
    if err and isinstance(err, dict):
        message = str(err.get("message") or "")
        if "column" in message and "builder_version" in message:
            payload.pop("builder_version", None)
            res = supabase.table("report_runs").insert(payload).execute()
            err = _res_error(res)
        if err and isinstance(err, dict):
            message = str(err.get("message") or "")
            if "column" in message and "customer_id" in message:
                payload.pop("customer_id", None)
                res = supabase.table("report_runs").insert(payload).execute()
                err = _res_error(res)
    if err:
        raise RuntimeError(f"Insert report_runs failed: {err}")
    rows = _res_data(res)
    if not rows:
        raise RuntimeError("Insert report_runs returned no rows.")
    return str(rows[0].get("id") or "")


def _insert_artifact(*, report_run_id: str, row: Dict[str, Any]) -> str:
    payload = {
        "session_id": str(report_run_id),
        "turn_id": int(row.get("turn_id") or 1),
        "artifact_type": str(row.get("artifact_type") or ""),
        "title": row.get("title"),
        "created_mode": row.get("created_mode") or "report_builder_v0",
        "source_tool_run_id": row.get("source_tool_run_id"),
        "source_tool_name": row.get("source_tool_name"),
        "format_spec": row.get("format_spec") or {},
        "payload": row.get("payload") or {},
        "row_count": row.get("row_count"),
        "bytes": row.get("bytes"),
    }
    res = supabase.table("artifacts").insert(payload).execute()
    err = _res_error(res)
    if err:
        raise RuntimeError(f"Insert artifacts failed: {err}")
    rows = _res_data(res)
    if not rows:
        raise RuntimeError("Insert artifacts returned no rows.")
    return str(rows[0].get("id") or "")


def _build_placeholder_payload() -> Dict[str, Any]:
    return {"kind": "comment_placeholder", "status": "empty", "text": None, "generated_by": None}


@log_tool_call("income_statement_tool")
def _pnl_tool(req: PnlRequest) -> Dict[str, Any]:
    res = pnl_tables(**req.model_dump(exclude={"session_id", "turn_id", "format_request"}))
    table_df = res["table"]
    return {
        "meta": res.get("meta"),
        "columns": list(table_df.columns),
        "table": _df_to_records(table_df),
    }


@log_tool_call("variance_tool")
def _variance_tool(req: VarianceRequest) -> Dict[str, Any]:
    res = variance_tables(**req.model_dump(exclude={"session_id", "turn_id", "format_request"}))
    out: Dict[str, Any] = {}
    for k, v in res.items():
        out[k] = _df_to_records(v) if hasattr(v, "to_dict") else v
    return out


def _run_income_statement(
    *,
    period: str,
    rows: List[str],
    compare_mode: str,
    module_id: str,
    visibility: str,
    module_index: int,
    session_id: str,
    turn_id: int,
) -> Tuple[str, str, int]:
    req = PnlRequest(
        session_id=session_id,
        turn_id=turn_id,
        compare_mode=compare_mode,
        periods=[period],
        rows=rows,
        filters=None,
        include_total=True,
    )
    resp = _pnl_tool(req)
    meta = resp.get("meta") if isinstance(resp.get("meta"), dict) else {}
    tool_run_id = str(meta.get("tool_run_id") or "")
    presentation_artifact_id = str(meta.get("presentation_artifact_id") or "")
    if not presentation_artifact_id:
        raise RuntimeError("Missing presentation_artifact_id from income_statement_tool")
    return tool_run_id, presentation_artifact_id, turn_id


def _prev_month(period: str) -> str:
    year_s, month_s = period.split("-")
    year = int(year_s)
    month = int(month_s)
    if month == 1:
        return f"{year - 1}-12"
    return f"{year}-{month - 1:02d}"


def _prev_year(period: str) -> str:
    year_s, month_s = period.split("-")
    year = int(year_s) - 1
    return f"{year}-{month_s}"


def _run_variance(
    *,
    period: str,
    settings: Dict[str, Any],
    module_id: str,
    visibility: str,
    module_index: int,
    session_id: str,
    turn_id: int,
) -> Tuple[str, str, int]:
    compare_mode = str(settings.get("compareMode") or "month").lower()
    base_period_mode = str(settings.get("basePeriod") or "lm").lower()
    grain = settings.get("grain") if isinstance(settings.get("grain"), list) else ["rr_level_2"]
    filters = settings.get("filters") if isinstance(settings.get("filters"), dict) else {}
    top_n = settings.get("topN")
    top_n_val = int(top_n) if isinstance(top_n, (int, float)) else None

    base_period = _prev_month(period) if base_period_mode == "lm" else _prev_year(period)
    req = VarianceRequest(
        session_id=session_id,
        turn_id=turn_id,
        compare_mode=compare_mode,
        base_period=base_period,
        comp_period=period,
        grain=[str(x) for x in grain],
        filters={k: v for k, v in filters.items() if v},
        top_n_pos=top_n_val,
        top_n_neg=top_n_val,
    )
    resp = _variance_tool(req)
    meta = resp.get("meta") if isinstance(resp.get("meta"), dict) else {}
    tool_run_id = str(meta.get("tool_run_id") or "")
    presentation_artifact_id = str(meta.get("presentation_artifact_id") or "")
    if not presentation_artifact_id:
        raise RuntimeError("Missing presentation_artifact_id from variance_tool")
    return tool_run_id, presentation_artifact_id, turn_id


def build_report_tables(request: Dict[str, Any]) -> Dict[str, Any]:
    period = str(request.get("period"))
    modules = list(request.get("modules") or [])
    module_configs = request.get("module_configs") if isinstance(request.get("module_configs"), dict) else {}
    customer_id = request.get("customer_id")
    format_overrides = request.get("format_overrides") if isinstance(request.get("format_overrides"), dict) else {}
    report_spec = {
        "builder_version": "report_builder_v0",
        "period": period,
        "customer_id": customer_id,
        "modules": modules,
        "module_configs": module_configs,
        "format_overrides": format_overrides,
        "notes": {"ui": "Rapportskapare Build Tables"},
    }

    report_run_id = _insert_report_run(period=period, report_spec=report_spec, customer_id=customer_id)
    module_results: List[Dict[str, Any]] = []
    artifact_count = 0
    tool_turn_id = 1

    for idx, module in enumerate(modules):
        module_instance_id = ""
        module_type = ""
        if isinstance(module, dict):
            module_instance_id = str(module.get("id") or "")
            module_type = str(module.get("type") or "")
        elif isinstance(module, str):
            module_instance_id = module
            module_type = module

        if not module_instance_id:
            module_instance_id = f"module-{idx}"

        if not module_type:
            module_type = module_instance_id

        module_status = "ok"
        warnings: List[str] = []
        visible_tables: List[Dict[str, Any]] = []
        supporting_tables: List[Dict[str, Any]] = []
        comment_placeholder: Optional[Dict[str, Any]] = None

        try:
            cfg = module_configs.get(module_instance_id) if isinstance(module_configs, dict) else {}
            if isinstance(cfg, dict) and "settings" in cfg:
                settings = cfg.get("settings") if isinstance(cfg.get("settings"), dict) else {}
                cfg_type = str(cfg.get("type") or "")
                if cfg_type:
                    module_type = cfg_type
            else:
                settings = cfg if isinstance(cfg, dict) else {}

            if module_type == "executive_summary":
                placeholder_payload = _build_placeholder_payload()
                placeholder_payload["meta"] = {
                    "module_id": module_instance_id,
                    "module_type": module_type,
                    "visibility": "placeholder",
                    "module_index": idx,
                }
                placeholder_id = _insert_artifact(
                    report_run_id=report_run_id,
                    row={
                        "artifact_type": "report_comment_placeholder",
                        "title": "Executive summary",
                        "payload": placeholder_payload,
                        "row_count": 0,
                        "bytes": _estimate_bytes(placeholder_payload),
                    },
                )
                artifact_count += 1
                comment_placeholder = {
                    "placeholder_id": placeholder_id,
                    "status": "empty",
                    "text": None,
                    "payload": placeholder_payload,
                }
            elif module_type == "income_statement":
                compare_mode = "ytd" if str(settings.get("periodMode") or "").lower() == "ytd" else "month"

                # Visible table
                tool_run_id, presentation_artifact_id, tool_turn_id_local = _run_income_statement(
                    period=period,
                    rows=["konto_typ", "rr_level_1", "rr_level_2"],
                    compare_mode=compare_mode,
                    module_id=module_instance_id,
                    visibility="visible",
                    module_index=idx,
                    session_id=report_run_id,
                    turn_id=tool_turn_id,
                )
                tool_turn_id += 1
                payload = {
                    "kind": "presentation_ref",
                    "presentation_artifact_id": presentation_artifact_id,
                    "meta": {
                        "module_id": module_instance_id,
                        "module_type": module_type,
                        "visibility": "visible",
                        "module_index": idx,
                        "source_tool_run_id": tool_run_id,
                        "presentation_artifact_id": presentation_artifact_id,
                        "tool_turn_id": tool_turn_id_local,
                    },
                }
                visible_id = _insert_artifact(
                    report_run_id=report_run_id,
                    row={
                        "artifact_type": "report_table",
                        "title": "Resultaträkning",
                        "payload": payload,
                        "source_tool_name": "income_statement_tool",
                        "source_tool_run_id": tool_run_id,
                        "row_count": None,
                        "bytes": _estimate_bytes(payload),
                    },
                )
                artifact_count += 1
                visible_tables.append(
                    {
                        "artifact_id": visible_id,
                        "title": "Resultaträkning",
                        "presentation_artifact_id": presentation_artifact_id,
                        "source_tool_run_id": tool_run_id,
                        "source_tool_name": "income_statement_tool",
                        "tool_turn_id": tool_turn_id_local,
                    }
                )

                # Supporting tables
                supporting_specs = [
                    (["konto_typ", "rr_level_1", "rr_level_2", "account"], "Resultaträkning (konto)"),
                    (["konto_typ", "rr_level_1", "rr_level_2", "unit"], "Resultaträkning (unit)"),
                ]
                for rows, title in supporting_specs:
                    tool_run_id, presentation_artifact_id, tool_turn_id_local = _run_income_statement(
                        period=period,
                        rows=rows,
                        compare_mode=compare_mode,
                        module_id=module_instance_id,
                        visibility="supporting",
                        module_index=idx,
                        session_id=report_run_id,
                        turn_id=tool_turn_id,
                    )
                    tool_turn_id += 1
                    sup_payload = {
                        "kind": "presentation_ref",
                        "presentation_artifact_id": presentation_artifact_id,
                        "meta": {
                            "module_id": module_instance_id,
                            "module_type": module_type,
                            "visibility": "supporting",
                            "module_index": idx,
                            "source_tool_run_id": tool_run_id,
                            "presentation_artifact_id": presentation_artifact_id,
                            "tool_turn_id": tool_turn_id_local,
                        },
                    }
                    sup_id = _insert_artifact(
                        report_run_id=report_run_id,
                        row={
                            "artifact_type": "report_table",
                            "title": title,
                            "payload": sup_payload,
                            "source_tool_name": "income_statement_tool",
                            "source_tool_run_id": tool_run_id,
                            "row_count": None,
                            "bytes": _estimate_bytes(sup_payload),
                        },
                    )
                    artifact_count += 1
                    supporting_tables.append(
                        {
                            "artifact_id": sup_id,
                            "title": title,
                            "presentation_artifact_id": presentation_artifact_id,
                            "source_tool_run_id": tool_run_id,
                            "source_tool_name": "income_statement_tool",
                            "tool_turn_id": tool_turn_id_local,
                        }
                    )

                placeholder_payload = _build_placeholder_payload()
                placeholder_payload["meta"] = {
                    "module_id": module_instance_id,
                    "module_type": module_type,
                    "visibility": "placeholder",
                    "module_index": idx,
                }
                placeholder_id = _insert_artifact(
                    report_run_id=report_run_id,
                    row={
                        "artifact_type": "report_comment_placeholder",
                        "title": "Resultaträkning",
                        "payload": placeholder_payload,
                        "row_count": 0,
                        "bytes": _estimate_bytes(placeholder_payload),
                    },
                )
                artifact_count += 1
                comment_placeholder = {
                    "placeholder_id": placeholder_id,
                    "status": "empty",
                    "text": None,
                    "payload": placeholder_payload,
                }
            elif module_type == "variance":
                tool_run_id, presentation_artifact_id, tool_turn_id_local = _run_variance(
                    period=period,
                    settings=settings,
                    module_id=module_instance_id,
                    visibility="visible",
                    module_index=idx,
                    session_id=report_run_id,
                    turn_id=tool_turn_id,
                )
                tool_turn_id += 1
                payload = {
                    "kind": "presentation_ref",
                    "presentation_artifact_id": presentation_artifact_id,
                    "meta": {
                        "module_id": module_instance_id,
                        "module_type": module_type,
                        "visibility": "visible",
                        "module_index": idx,
                        "source_tool_run_id": tool_run_id,
                        "presentation_artifact_id": presentation_artifact_id,
                        "tool_turn_id": tool_turn_id_local,
                    },
                }
                visible_id = _insert_artifact(
                    report_run_id=report_run_id,
                    row={
                        "artifact_type": "report_table",
                        "title": "Variansanalys",
                        "payload": payload,
                        "source_tool_name": "variance_tool",
                        "source_tool_run_id": tool_run_id,
                        "row_count": None,
                        "bytes": _estimate_bytes(payload),
                    },
                )
                artifact_count += 1
                visible_tables.append(
                    {
                        "artifact_id": visible_id,
                        "title": "Variansanalys",
                        "presentation_artifact_id": presentation_artifact_id,
                        "source_tool_run_id": tool_run_id,
                        "source_tool_name": "variance_tool",
                        "tool_turn_id": tool_turn_id_local,
                    }
                )
            else:
                module_status = "warn"
                warnings.append(f"Module not supported in V0: {module_type}")
        except Exception as exc:
            module_status = "error"
            warnings.append(f"{type(exc).__name__}: {exc}")

        module_results.append(
            {
                "module_id": module_instance_id,
                "visible_tables": visible_tables,
                "supporting_tables": supporting_tables,
                "comment_placeholder": comment_placeholder,
                "module_status": module_status,
                "warnings": warnings,
            }
        )

    logger.info("report_builder: report_run_id=%s artifacts=%s", report_run_id, artifact_count)
    return {"report_run_id": report_run_id, "status": "ok", "modules": module_results, "report_spec": report_spec}


def fetch_report_run(*, report_run_id: str) -> Dict[str, Any]:
    run_res = (
        supabase.table("report_runs")
        .select("id,report_spec")
        .eq("id", report_run_id)
        .single()
        .execute()
    )
    run_row = run_res.data if isinstance(getattr(run_res, "data", None), dict) else {}
    report_spec = run_row.get("report_spec") if isinstance(run_row, dict) else None
    overrides = {}
    if isinstance(report_spec, dict):
        overrides = report_spec.get("table_overrides") if isinstance(report_spec.get("table_overrides"), dict) else {}
    res = (
        supabase
        .table("artifacts")
        .select("id,artifact_type,title,payload,source_tool_name,created_at")
        .eq("session_id", report_run_id)
        .order("created_at", desc=False)
        .execute()
    )
    rows = _res_data(res)
    modules_map: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        artifact_type = str(row.get("artifact_type") or "")
        if artifact_type not in {"report_table", "report_comment_placeholder"}:
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        module_id = str(meta.get("module_id") or "")
        if not module_id:
            continue
        module_index = int(meta.get("module_index") or 0)
        visibility = str(meta.get("visibility") or "")

        mod = modules_map.setdefault(
            module_id,
            {
                "module_id": module_id,
                "visible_tables": [],
                "supporting_tables": [],
                "comment_placeholder": None,
                "module_status": "ok",
                "warnings": [],
                "_module_index": module_index,
            },
        )
        mod["_module_index"] = min(mod.get("_module_index", module_index), module_index)

        if artifact_type == "report_comment_placeholder":
            mod["comment_placeholder"] = {
                "placeholder_id": row.get("id"),
                "status": "empty",
                "text": None,
                "payload": payload,
            }
            continue

        presentation_artifact_id = payload.get("presentation_artifact_id") if isinstance(payload, dict) else None
        if not presentation_artifact_id and isinstance(meta, dict):
            presentation_artifact_id = meta.get("presentation_artifact_id")
        override = overrides.get(str(row.get("id"))) if isinstance(overrides, dict) else None
        if isinstance(override, dict):
            override_id = override.get("presentation_artifact_id")
            if isinstance(override_id, str) and override_id:
                presentation_artifact_id = override_id
        entry = {
            "artifact_id": row.get("id"),
            "title": row.get("title"),
            "presentation_artifact_id": presentation_artifact_id,
            "source_tool_run_id": meta.get("source_tool_run_id"),
            "source_tool_name": row.get("source_tool_name"),
            "tool_turn_id": meta.get("tool_turn_id"),
        }
        if visibility == "supporting":
            mod["supporting_tables"].append(entry)
        else:
            mod["visible_tables"].append(entry)

    modules_sorted = sorted(
        [v for v in modules_map.values()],
        key=lambda x: int(x.get("_module_index", 0)),
    )
    for mod in modules_sorted:
        mod.pop("_module_index", None)

    return {
        "report_run_id": report_run_id,
        "status": "ok",
        "report_spec": report_spec,
        "modules": modules_sorted,
    }


def fetch_presentation_artifacts(*, artifact_ids: List[str]) -> List[Dict[str, Any]]:
    ids = [str(x) for x in (artifact_ids or []) if str(x).strip()]
    if not ids:
        return []
    res = (
        supabase
        .table("artifacts")
        .select("id,title,artifact_type,payload,format_spec,source_tool_run_id,source_tool_name")
        .in_("id", ids)
        .execute()
    )
    rows = _res_data(res)
    out = []
    for row in rows:
        if row.get("artifact_type") != "presentation_table":
            continue
        out.append(row)
    return out


def _is_numeric_column(col_key: str, rows: List[Dict[str, Any]]) -> bool:
    for row in rows:
        val = row.get(col_key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return True
    return False


def _apply_row_grouping(
    *,
    rows: List[Dict[str, Any]],
    columns: List[str],
    presentation_artifact_id: str,
    grouping_spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    groups = grouping_spec.get("groups") if isinstance(grouping_spec, dict) else None
    if not groups:
        return rows

    row_lookup: Dict[str, Dict[str, Any]] = {}
    row_index: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        row_id = f"{presentation_artifact_id}:{idx}"
        row_lookup[row_id] = row
        row_index[row_id] = idx

    grouped_member_ids: set[str] = set()
    for group in groups:
        member_ids = group.get("memberRowIds") if isinstance(group, dict) else None
        if isinstance(member_ids, list):
            grouped_member_ids.update([str(x) for x in member_ids])

    group_rows_by_first_member: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        member_ids = group.get("memberRowIds")
        if not isinstance(member_ids, list):
            continue
        member_rows: List[Tuple[str, Dict[str, Any]]] = []
        for member_id in member_ids:
            member_id_str = str(member_id)
            row = row_lookup.get(member_id_str)
            if row is not None:
                member_rows.append((member_id_str, row))
        if not member_rows:
            continue

        first_member_id = min(member_rows, key=lambda item: row_index.get(item[0], 0))[0]
        labels = group.get("labels") if isinstance(group.get("labels"), dict) else {}

        group_row: Dict[str, Any] = {
            "__rowId": f"group:{group.get('groupId')}",
            "__source": "group",
            "__groupId": group.get("groupId"),
        }
        for col in columns:
            if _is_numeric_column(col, [r for _, r in member_rows]):
                total = 0
                for _, row in member_rows:
                    val = row.get(col)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        total += val
                group_row[col] = total
            else:
                group_row[col] = labels.get(col, "")

        group_rows_by_first_member.setdefault(first_member_id, []).append(group_row)

    hide_members = grouping_spec.get("hideMembers", True)
    output: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        row_id = f"{presentation_artifact_id}:{idx}"
        group_rows = group_rows_by_first_member.get(row_id)
        if group_rows:
            output.extend(group_rows)
        if hide_members and row_id in grouped_member_ids:
            continue
        output.append(row)

    return output


def materialize_table_view(
    *,
    report_run_id: str,
    report_table_id: str,
    presentation_artifact_id: str,
    module_id: str,
    view_spec: Dict[str, Any],
    visibility: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    res = (
        supabase
        .table("artifacts")
        .select("id,title,artifact_type,payload,format_spec,source_tool_run_id,source_tool_name")
        .eq("id", presentation_artifact_id)
        .limit(1)
        .execute()
    )
    rows = _res_data(res)
    if not rows:
        raise RuntimeError("presentation_table not found")
    source_artifact = rows[0]
    if source_artifact.get("artifact_type") != "presentation_table":
        raise RuntimeError("Source artifact is not presentation_table")

    payload = source_artifact.get("payload") if isinstance(source_artifact.get("payload"), dict) else {}
    columns = payload.get("columns") if isinstance(payload.get("columns"), list) else []
    columns = [str(c) for c in columns]
    raw_rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    table_rows = [r for r in raw_rows if isinstance(r, dict)]

    grouping_spec = view_spec.get("rowGrouping") if isinstance(view_spec, dict) else None
    grouped_rows = _apply_row_grouping(
        rows=table_rows,
        columns=columns,
        presentation_artifact_id=presentation_artifact_id,
        grouping_spec=grouping_spec or {},
    )

    next_payload = dict(payload)
    next_payload["kind"] = "table"
    next_payload["columns"] = columns
    next_payload["rows"] = grouped_rows
    meta = next_payload.get("meta") if isinstance(next_payload.get("meta"), dict) else {}
    meta.update({
        "module_id": module_id,
        "visibility": visibility or meta.get("visibility"),
        "source_presentation_artifact_id": presentation_artifact_id,
        "view_spec": view_spec,
    })
    next_payload["meta"] = meta

    new_artifact_id = _insert_artifact(
        report_run_id=report_run_id,
        row={
            "artifact_type": "presentation_table",
            "title": title or source_artifact.get("title"),
            "payload": next_payload,
            "source_tool_name": source_artifact.get("source_tool_name"),
            "source_tool_run_id": source_artifact.get("source_tool_run_id"),
            "format_spec": source_artifact.get("format_spec") or {},
            "created_mode": "report_builder_view_materialize",
            "row_count": len(grouped_rows),
            "bytes": _estimate_bytes(next_payload),
        },
    )

    run_res = (
        supabase
        .table("report_runs")
        .select("report_spec")
        .eq("id", report_run_id)
        .single()
        .execute()
    )
    run_row = run_res.data if isinstance(getattr(run_res, "data", None), dict) else {}
    report_spec = run_row.get("report_spec") if isinstance(run_row, dict) else {}
    if not isinstance(report_spec, dict):
        report_spec = {}
    overrides = report_spec.get("table_overrides") if isinstance(report_spec.get("table_overrides"), dict) else {}
    overrides[str(report_table_id)] = {
        "presentation_artifact_id": new_artifact_id,
        "view_spec": view_spec,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_spec["table_overrides"] = overrides
    supabase.table("report_runs").update({"report_spec": report_spec}).eq("id", report_run_id).execute()

    return new_artifact_id
