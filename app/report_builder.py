from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from .supabase_client import supabase
from .tools.income_statement_tool import pnl_tables
from .tools.variance_tool import variance_tables
from .schemas.pnl import PnlRequest
from .schemas.variance import VarianceRequest
from .tool_logging import log_tool_call
from .settings import settings

logger = logging.getLogger(__name__)


def _res_data(res: Any) -> List[Dict[str, Any]]:
    data = getattr(res, "data", None)
    if data is None:
        data = res.data
    return data or []


def _res_error(res: Any) -> Any:
    return getattr(res, "error", None)


def _select_single(query: Any) -> Any:
    """
    Supabase client compatibility: use .single() when available, otherwise .limit(1).
    """
    if hasattr(query, "single"):
        try:
            return query.single()
        except Exception:
            return query.limit(1)
    return query.limit(1)


def _first_row(res: Any) -> Dict[str, Any]:
    data = getattr(res, "data", None)
    if data is None:
        data = res.data
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], dict) else {}
    return {}


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
    return {
        "kind": "comment_placeholder",
        "status": "empty",
        "text": None,
        "generated_by": None,
        "reviewer_summary": None,
        "evidence_refs": None,
    }


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
                        "title": "Variansanalys",
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
    run_query = (
        supabase.table("report_runs")
        .select("id,report_spec")
        .eq("id", report_run_id)
    )
    run_res = _select_single(run_query).execute()
    run_row = _first_row(run_res)
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
            status = str(payload.get("status") or "empty") if isinstance(payload, dict) else "empty"
            text = payload.get("text") if isinstance(payload, dict) else None
            reviewer_summary = payload.get("reviewer_summary") if isinstance(payload, dict) else None
            evidence_refs = payload.get("evidence_refs") if isinstance(payload, dict) else None
            mod["comment_placeholder"] = {
                "placeholder_id": row.get("id"),
                "status": status,
                "text": text,
                "reviewer_summary": reviewer_summary,
                "evidence_refs": evidence_refs,
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
            "payload": payload,
        }
        if visibility in {"supporting", "supporting_auto"}:
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


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _get_openai_cfg() -> Tuple[str, str]:
    base_url = (os.environ.get("OPENAI_BASE_URL") or settings.OPENAI_BASE_URL or "https://api.openai.com").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return base_url, api_key


def _openai_model(default: Optional[str] = None) -> str:
    return default or os.environ.get("OPENAI_MODEL") or settings.OPENAI_MODEL or "gpt-4o-mini"


def _call_openai_json(
    *,
    system: str,
    user_payload: Dict[str, Any],
    model: Optional[str] = None,
    temperature: float = 0,
    timeout_s: int = 30,
) -> Dict[str, Any]:
    base_url, api_key = _get_openai_cfg()
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": _openai_model(model),
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _safe_json(user_payload)},
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
    try:
        out = json.loads(content)
    except Exception:
        raise RuntimeError(f"LLM did not return valid JSON. content={content[:300]!r}")
    if not isinstance(out, dict):
        raise RuntimeError("LLM JSON must be an object")
    return out


def _row_label(row: Dict[str, Any], columns: List[str]) -> Optional[str]:
    for col in columns:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _guess_totals_marker(payload: Dict[str, Any]) -> str:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    marker = meta.get("totals_marker") if isinstance(meta, dict) else None
    return str(marker) if marker else "__TOTAL__"


def _is_total_row(row: Dict[str, Any], columns: List[str], totals_marker: str) -> bool:
    for col in columns:
        val = row.get(col)
        if val is None:
            continue
        v = str(val).strip()
        if not v:
            continue
        if v.upper() in {"TOTAL", "SUMMA", totals_marker}:
            return True
    return False


def _select_highlight_rows(
    *,
    rows: List[Dict[str, Any]],
    columns: List[str],
    presentation_artifact_id: str,
    row_id_prefix: Optional[str] = None,
    totals_marker: str,
    max_rows: int = 20,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    if not rows or not columns:
        return []
    primary_numeric: Optional[str] = None
    for col in reversed(columns):
        if _is_numeric_column(col, rows):
            primary_numeric = col
            break

    indexed_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        prefix = row_id_prefix or presentation_artifact_id
        row_id = str(row.get("__rowId") or f"{prefix}:{idx}")
        indexed_rows.append(
            {
                "row_id": row_id,
                "label": _row_label(row, columns),
                "values": {col: row.get(col) for col in columns},
                "_raw": row,
            }
        )

    totals = [r for r in indexed_rows if _is_total_row(r["_raw"], columns, totals_marker)]
    first_rows = indexed_rows[: min(5, len(indexed_rows))]
    top_rows: List[Dict[str, Any]] = []
    if primary_numeric:
        sorted_rows = sorted(
            indexed_rows,
            key=lambda r: abs(r["_raw"].get(primary_numeric) or 0)
            if isinstance(r["_raw"].get(primary_numeric), (int, float)) and not isinstance(r["_raw"].get(primary_numeric), bool)
            else 0,
            reverse=True,
        )
        top_rows = sorted_rows[: min(top_n, len(sorted_rows))]

    merged: Dict[str, Dict[str, Any]] = {}
    for r in totals + top_rows + first_rows:
        merged[r["row_id"]] = {k: v for k, v in r.items() if k != "_raw"}
    out = list(merged.values())
    return out[:max_rows]


def _presentation_table_evidence(
    *,
    report_table_id: str,
    presentation_artifact: Dict[str, Any],
    visibility: str,
    title: Optional[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    payload = presentation_artifact.get("payload") if isinstance(presentation_artifact.get("payload"), dict) else {}
    if payload.get("kind") == "multi_table" and isinstance(payload.get("tables"), list):
        out: List[Dict[str, Any]] = []
        for table_payload in payload.get("tables") or []:
            if not isinstance(table_payload, dict):
                continue
            columns = table_payload.get("columns") if isinstance(table_payload.get("columns"), list) else []
            columns = [str(c) for c in columns]
            rows = table_payload.get("rows") if isinstance(table_payload.get("rows"), list) else []
            table_rows = [r for r in rows if isinstance(r, dict)]
            totals_marker = _guess_totals_marker(table_payload)
            table_key = str(table_payload.get("table_key") or "")
            row_id_prefix = f"{presentation_artifact.get('id')}:{table_key}" if table_key else str(presentation_artifact.get("id") or "")
            highlighted_rows = _select_highlight_rows(
                rows=table_rows,
                columns=columns,
                presentation_artifact_id=str(presentation_artifact.get("id") or ""),
                row_id_prefix=row_id_prefix,
                totals_marker=totals_marker,
                max_rows=max_rows,
            )
            out.append(
                {
                    "report_table_id": report_table_id,
                    "presentation_artifact_id": str(presentation_artifact.get("id") or ""),
                    "title": title,
                    "visibility": visibility,
                    "table_key": table_key or None,
                    "columns": columns,
                    "rows": highlighted_rows,
                    "row_count": len(table_rows),
                    "totals_marker": totals_marker,
                }
            )
        return out
    columns = payload.get("columns") if isinstance(payload.get("columns"), list) else []
    columns = [str(c) for c in columns]
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    table_rows = [r for r in rows if isinstance(r, dict)]
    totals_marker = _guess_totals_marker(payload)
    highlighted_rows = _select_highlight_rows(
        rows=table_rows,
        columns=columns,
        presentation_artifact_id=str(presentation_artifact.get("id") or ""),
        totals_marker=totals_marker,
        max_rows=max_rows,
    )
    return [
        {
            "report_table_id": report_table_id,
            "presentation_artifact_id": str(presentation_artifact.get("id") or ""),
            "title": title,
            "visibility": visibility,
            "table_key": None,
            "columns": columns,
            "rows": highlighted_rows,
            "row_count": len(table_rows),
            "totals_marker": totals_marker,
        }
    ]


def _resolve_module_definitions(report_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    modules = list(report_spec.get("modules") or [])
    module_configs = report_spec.get("module_configs") if isinstance(report_spec.get("module_configs"), dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for idx, module in enumerate(modules):
        module_id = ""
        module_type = ""
        if isinstance(module, dict):
            module_id = str(module.get("id") or "")
            module_type = str(module.get("type") or "")
        elif isinstance(module, str):
            module_id = module
            module_type = module
        if not module_id:
            module_id = f"module-{idx}"
        if not module_type:
            module_type = module_id
        cfg = module_configs.get(module_id) if isinstance(module_configs, dict) else {}
        settings: Dict[str, Any] = {}
        if isinstance(cfg, dict) and "settings" in cfg:
            settings = cfg.get("settings") if isinstance(cfg.get("settings"), dict) else {}
            cfg_type = str(cfg.get("type") or "")
            if cfg_type:
                module_type = cfg_type
        elif isinstance(cfg, dict):
            settings = cfg
        out[module_id] = {
            "module_id": module_id,
            "module_type": module_type,
            "module_index": idx,
            "settings": settings,
        }
    return out


def _update_artifact_payload(*, artifact_id: str, payload: Dict[str, Any]) -> None:
    supabase.table("artifacts").update({"payload": payload, "bytes": _estimate_bytes(payload)}).eq("id", artifact_id).execute()


def _writer_prompt(language: str) -> str:
    return (
        "You are a CFO/controller writing report comments.\n"
        "Rules:\n"
        "- Use ONLY the provided evidence tables. No hallucinations.\n"
        "- If evidence is insufficient, say so explicitly.\n"
        "- Keep a concise, professional tone.\n"
        "- Output MUST be valid JSON only (no markdown).\n"
        "Required JSON keys:\n"
        "- comment_markdown\n"
        "- claim_evidence: list of {report_table_id, presentation_artifact_id, row_ids[]}\n"
        "- what_changed\n"
        "- why\n"
        "- actions\n"
        "- questions\n"
        f"- language: {language}\n"
    )


def _reviewer_prompt(language: str, allowed_actions: List[str]) -> str:
    allowed_list = ", ".join(allowed_actions) if allowed_actions else "none"
    return (
        "You are a strict reviewer validating a finance report comment.\n"
        "Rules:\n"
        "- Verify every claim against evidence rows.\n"
        "- Flag unsupported or weak claims.\n"
        "- Ensure executive summary captures the biggest storylines.\n"
        "- Do NOT ask the user questions.\n"
        "- Output MUST be valid JSON only (no markdown).\n"
        "Required JSON keys:\n"
        "- verdict: approve|revise|needs_drilldown\n"
        "- issues: list of strings\n"
        "- suggested_rewrite: string or null\n"
        "- drilldown_actions: list of allowed actions only\n"
        f"- language: {language}\n"
        f"Allowed drilldown_actions: {allowed_list}\n"
    )


def _drilldown_allowlist(module_type: str) -> List[str]:
    if module_type == "income_statement":
        return ["breakdown_by_account", "breakdown_by_unit"]
    if module_type == "variance":
        return ["breakdown_by_account", "breakdown_by_unit", "top_drivers"]
    return []


def _run_drilldown_action(
    *,
    report_run_id: str,
    module_id: str,
    module_type: str,
    module_index: int,
    period: str,
    settings: Dict[str, Any],
    action: str,
    turn_id: int,
) -> Tuple[str, str, int, str]:
    if module_type == "income_statement":
        if action == "breakdown_by_account":
            rows = ["konto_typ", "rr_level_1", "rr_level_2", "account"]
            title = "Resultaträkning (konto, auto)"
        elif action == "breakdown_by_unit":
            rows = ["konto_typ", "rr_level_1", "rr_level_2", "unit"]
            title = "Resultaträkning (unit, auto)"
        else:
            raise RuntimeError(f"Unsupported drilldown action: {action}")
        compare_mode = "ytd" if str(settings.get("periodMode") or "").lower() == "ytd" else "month"
        tool_run_id, presentation_artifact_id, turn_id_local = _run_income_statement(
            period=period,
            rows=rows,
            compare_mode=compare_mode,
            module_id=module_id,
            visibility="supporting_auto",
            module_index=module_index,
            session_id=report_run_id,
            turn_id=turn_id,
        )
        return tool_run_id, presentation_artifact_id, turn_id_local, title

    if module_type == "variance":
        cfg = dict(settings or {})
        grain = cfg.get("grain") if isinstance(cfg.get("grain"), list) else ["rr_level_2"]
        if action == "breakdown_by_account":
            cfg["grain"] = [str(x) for x in (grain + ["account"]) if x]
            title = "Variansanalys (konto, auto)"
        elif action == "breakdown_by_unit":
            cfg["grain"] = [str(x) for x in (grain + ["unit"]) if x]
            title = "Variansanalys (unit, auto)"
        elif action == "top_drivers":
            cfg["topN"] = int(cfg.get("topN") or 10)
            title = "Variansanalys (top drivers, auto)"
        else:
            raise RuntimeError(f"Unsupported drilldown action: {action}")
        tool_run_id, presentation_artifact_id, turn_id_local = _run_variance(
            period=period,
            settings=cfg,
            module_id=module_id,
            visibility="supporting_auto",
            module_index=module_index,
            session_id=report_run_id,
            turn_id=turn_id,
        )
        return tool_run_id, presentation_artifact_id, turn_id_local, title

    raise RuntimeError(f"Unsupported module type for drilldown: {module_type}")


def generate_report_comments(
    *,
    report_run_id: str,
    max_rounds: int = 1,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    run = fetch_report_run(report_run_id=report_run_id)
    report_spec = run.get("report_spec") if isinstance(run.get("report_spec"), dict) else {}
    module_defs = _resolve_module_definitions(report_spec)
    period = str(report_spec.get("period") or "")
    lang = str(language or "sv")

    modules = run.get("modules") if isinstance(run.get("modules"), list) else []
    all_tables: List[Dict[str, Any]] = []
    for mod in modules:
        for t in mod.get("visible_tables") or []:
            all_tables.append({**t, "_visibility": "visible", "_module_id": mod.get("module_id")})
        for t in mod.get("supporting_tables") or []:
            all_tables.append({**t, "_visibility": "supporting", "_module_id": mod.get("module_id")})

    presentation_ids = [t.get("presentation_artifact_id") for t in all_tables if t.get("presentation_artifact_id")]
    presentation_rows = fetch_presentation_artifacts(artifact_ids=[str(x) for x in presentation_ids if x])
    presentation_map = {str(r.get("id")): r for r in presentation_rows if isinstance(r, dict)}

    for mod in modules:
        module_id = str(mod.get("module_id") or "")
        if not module_id:
            continue
        definition = module_defs.get(module_id) or {}
        module_type = str(definition.get("module_type") or "")
        if module_type not in {"executive_summary", "income_statement", "variance"}:
            continue
        placeholder = mod.get("comment_placeholder") if isinstance(mod.get("comment_placeholder"), dict) else None
        if not placeholder or not placeholder.get("placeholder_id"):
            continue

        # Evidence selection
        if module_type == "executive_summary":
            module_tables = [t for t in all_tables if t.get("_module_id") != module_id]
        else:
            module_tables = [
                t for t in all_tables if str(t.get("_module_id") or "") == module_id
            ]

        evidence_tables: List[Dict[str, Any]] = []
        for t in module_tables:
            pres_id = str(t.get("presentation_artifact_id") or "")
            pres = presentation_map.get(pres_id)
            if not pres:
                continue
            evidence_tables.extend(
                _presentation_table_evidence(
                    report_table_id=str(t.get("artifact_id") or ""),
                    presentation_artifact=pres,
                    visibility=str(t.get("_visibility") or "visible"),
                    title=str(t.get("title") or ""),
                    max_rows=20,
                )
            )

        evidence_bundle = {
            "module_id": module_id,
            "module_type": module_type,
            "tables": evidence_tables,
        }
        rounds = 0
        final_text: Optional[str] = None
        reviewer_summary: Optional[Dict[str, Any]] = None
        status = "filled"
        debug_flow: List[Dict[str, Any]] = []
        existing_turn_ids = [
            int(t.get("tool_turn_id") or 0)
            for t in module_tables
            if str(t.get("_module_id") or "") == module_id
        ]
        tool_turn_id = (max(existing_turn_ids) + 1) if existing_turn_ids else 1

        while True:
            writer_payload = {
                "module": {"id": module_id, "type": module_type},
                "language": lang,
                "evidence": evidence_bundle,
            }
            writer_out = _call_openai_json(
                system=_writer_prompt(lang),
                user_payload=writer_payload,
                model=model,
                temperature=0,
            )
            writer_text = writer_out.get("comment_markdown") or writer_out.get("text") or ""
            debug_flow.append({"round": rounds + 1, "role": "writer", "output": writer_out})

            reviewer_out = _call_openai_json(
                system=_reviewer_prompt(lang, _drilldown_allowlist(module_type)),
                user_payload={
                    "module": {"id": module_id, "type": module_type},
                    "language": lang,
                    "evidence": evidence_bundle,
                    "writer_output": writer_out,
                },
                model=model,
                temperature=0,
            )
            debug_flow.append({"round": rounds + 1, "role": "reviewer", "output": reviewer_out})

            verdict = str(reviewer_out.get("verdict") or "revise").lower()
            reviewer_summary = reviewer_out

            if verdict == "approve":
                final_text = str(writer_text).strip()
                status = "approved"
                break
            if verdict == "revise":
                suggested = reviewer_out.get("suggested_rewrite")
                writer_rewrite = _call_openai_json(
                    system=_writer_prompt(lang),
                    user_payload={
                        "module": {"id": module_id, "type": module_type},
                        "language": lang,
                        "evidence": evidence_bundle,
                        "writer_output": writer_out,
                        "reviewer_feedback": reviewer_out,
                        "suggested_rewrite": suggested,
                    },
                    model=model,
                    temperature=0,
                )
                debug_flow.append({"round": rounds + 1, "role": "writer_rewrite", "output": writer_rewrite})
                rewrite_text = writer_rewrite.get("comment_markdown") or writer_rewrite.get("text")
                if isinstance(rewrite_text, str) and rewrite_text.strip():
                    final_text = rewrite_text.strip()
                elif isinstance(suggested, str) and suggested.strip():
                    final_text = suggested.strip()
                else:
                    final_text = str(writer_text).strip()
                status = "filled"
                break
            if verdict == "needs_drilldown" and rounds < int(max_rounds or 0):
                actions = reviewer_out.get("drilldown_actions")
                if not isinstance(actions, list):
                    actions = []
                allowed = _drilldown_allowlist(module_type)
                actions = [str(a) for a in actions if str(a) in allowed][:2]
                if not actions:
                    final_text = str(writer_text).strip()
                    status = "filled"
                    break
                definition = module_defs.get(module_id) or {}
                settings = definition.get("settings") if isinstance(definition.get("settings"), dict) else {}
                for action in actions:
                    tool_run_id, presentation_artifact_id, turn_id_local, title = _run_drilldown_action(
                        report_run_id=report_run_id,
                        module_id=module_id,
                        module_type=module_type,
                        module_index=int(definition.get("module_index") or 0),
                        period=period,
                        settings=settings,
                        action=action,
                        turn_id=tool_turn_id,
                    )
                    tool_turn_id += 1
                    payload = {
                        "kind": "presentation_ref",
                        "presentation_artifact_id": presentation_artifact_id,
                        "meta": {
                            "module_id": module_id,
                            "module_type": module_type,
                            "visibility": "supporting_auto",
                            "module_index": int(definition.get("module_index") or 0),
                            "source_tool_run_id": tool_run_id,
                            "presentation_artifact_id": presentation_artifact_id,
                            "tool_turn_id": turn_id_local,
                        },
                    }
                    report_table_id = _insert_artifact(
                        report_run_id=report_run_id,
                        row={
                            "artifact_type": "report_table",
                            "title": title,
                            "payload": payload,
                            "source_tool_name": ("income_statement_tool" if module_type == "income_statement" else "variance_tool"),
                            "source_tool_run_id": tool_run_id,
                            "row_count": None,
                            "bytes": _estimate_bytes(payload),
                        },
                    )
                    # Update evidence bundle with new presentation artifact
                    pres_rows = fetch_presentation_artifacts(artifact_ids=[presentation_artifact_id])
                    if pres_rows:
                        presentation_map[str(presentation_artifact_id)] = pres_rows[0]
                        evidence_tables.extend(
                            _presentation_table_evidence(
                                report_table_id=report_table_id,
                                presentation_artifact=pres_rows[0],
                                visibility="supporting_auto",
                                title=title,
                                max_rows=20,
                            )
                        )
                evidence_bundle["tables"] = evidence_tables
                rounds += 1
                continue

            final_text = str(writer_text).strip()
            status = "filled"
            break

        evidence_refs = [
            {
                "report_table_id": e.get("report_table_id"),
                "presentation_artifact_id": e.get("presentation_artifact_id"),
                "visibility": e.get("visibility"),
                "table_key": e.get("table_key"),
                "row_ids": [r.get("row_id") for r in (e.get("rows") or []) if isinstance(r, dict)],
            }
            for e in (evidence_bundle.get("tables") or [])
        ]
        placeholder_payload = placeholder.get("payload") if isinstance(placeholder.get("payload"), dict) else {}
        placeholder_payload.update(
            {
                "status": status,
                "text": final_text,
                "generated_by": {
                    "model": _openai_model(model),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "evidence_refs": evidence_refs,
                    "rounds": rounds,
                },
                "reviewer_summary": reviewer_summary,
                "evidence_refs": evidence_refs,
                "debug_flow": debug_flow,
            }
        )
        _update_artifact_payload(artifact_id=str(placeholder.get("placeholder_id")), payload=placeholder_payload)

    return fetch_report_run(report_run_id=report_run_id)


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

    run_query = (
        supabase
        .table("report_runs")
        .select("report_spec")
        .eq("id", report_run_id)
    )
    run_res = _select_single(run_query).execute()
    run_row = _first_row(run_res)
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
