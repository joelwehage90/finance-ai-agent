# tool_logging.py
from __future__ import annotations

import functools
import inspect
import json
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from .supabase_client import supabase
from .formatting.format_spec import default_format_spec
from .formatting.format_summary import build_format_summary_sv
from .formatting.presentation_table import (
    build_presentation_table_payload_from_tool_run,
    upsert_singleton_presentation_artifact,
)
from .formatting.spec_pipeline import resolve_incremental_format_spec

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Auto-format defaults: create a presentation_table artifact after every tool call that returns a table-ish payload.
# (Best-effort: failures here must never break the tool response.)
AUTO_FORMAT_TOOL_NAMES = {
    "income_statement_tool",
    "variance_tool",
    "definitions_tool",
    "account_mapping_tool",
    "nl_sql_tool",
}


def _to_jsonable(obj: Any) -> Any:
    """
    Legacy fallback (keep), but we primarily use FastAPI's jsonable_encoder now.
    This remains useful for extremely odd objects that slip through.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


def _estimate_bytes(payload: Any) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except Exception:
        # Fallback attempt
        try:
            return len(json.dumps(_to_jsonable(payload), ensure_ascii=False).encode("utf-8"))
        except Exception:
            return 0


def _infer_row_count(resp: Any) -> Optional[int]:
    # Standard: {"table":[...]} -> row_count = len(table)
    if isinstance(resp, dict) and isinstance(resp.get("table"), list):
        return len(resp["table"])
    return None


def _res_data(res: Any) -> List[Dict[str, Any]]:
    data = getattr(res, "data", None)
    if data is None:
        data = res.data
    return data or []


def _res_error(res: Any) -> Any:
    return getattr(res, "error", None)


def _insert_tool_run(payload: Dict[str, Any]) -> str:
    """
    Insert a final tool_run row.

    NOTE: Your `tool_runs` table has a CHECK constraint `tool_runs_status_check` that does NOT allow
    a transient 'running' status. So we only ever insert terminal rows: 'success' or 'error'.
    """
    res = supabase.table("tool_runs").insert(payload).execute()

    err = _res_error(res)
    if err:
        raise RuntimeError(f"Insert tool_runs failed: {err}")

    rows = _res_data(res)
    if not rows:
        raise RuntimeError("Insert tool_runs returned no rows (likely permissions/RLS).")

    return rows[0]["id"]


def _try_insert_tool_run(payload: Dict[str, Any], *, tool_name: str) -> Optional[str]:
    """
    Best-effort tool_runs insert.

    IMPORTANT:
    - Tool logging must never break the tool response in production.
    - Supabase/PostgREST can occasionally time out (e.g. Cloudflare 522). In those cases,
      we prefer to return the tool response without a tool_run_id.
    """
    try:
        return _insert_tool_run(payload)
    except Exception as e:
        logger.exception("tool_runs insert failed for %s (best-effort): %s", tool_name, e)
        return None


def _build_error_json(exc: Exception) -> Dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def log_tool_call(tool_name: str):
    """
    FastAPI-safe decorator:
    - Preserves original endpoint signature (avoids args/kwargs treated as query params).
    - Logs request/response into tool_runs using session_id + turn_id.
    - Ensures response_json is JSONB-safe via jsonable_encoder.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        sig = inspect.signature(fn)

        def _find_req_obj(args: Any, kwargs: Any) -> BaseModel:
            # Look through kwargs first, then args
            for v in list(kwargs.values()) + list(args):
                if isinstance(v, BaseModel) and hasattr(v, "session_id") and hasattr(v, "turn_id"):
                    return v
            raise RuntimeError(f"{tool_name}: request model with session_id/turn_id not found.")

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                req_obj = _find_req_obj(args, kwargs)

                # request payload as json-safe
                req_dict = req_obj.model_dump(mode="json")
                session_id = req_dict.get("session_id")
                turn_id = req_dict.get("turn_id")
                if not session_id or turn_id is None:
                    raise ValueError(f"{tool_name}: session_id and turn_id are required.")

                started = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    duration_ms = int((time.perf_counter() - started) * 1000)

                    # Ensure json-safe and include tool_run_id in meta of returned payload (so agent can reference)
                    safe_result = jsonable_encoder(result) if result is not None else None
                    safe_request = jsonable_encoder(req_dict)
                    tool_run_id = _try_insert_tool_run(
                        {
                            "session_id": str(session_id),
                            "turn_id": int(turn_id),
                            "tool_name": tool_name,
                            "status": "success",
                            "request_json": safe_request,
                            "response_json": safe_result,
                            "duration_ms": duration_ms,
                            "row_count": _infer_row_count(safe_result),
                            "bytes": _estimate_bytes(safe_result),
                        },
                        tool_name=tool_name,
                    )

                    # Auto-create formatted/presentation artifact (best-effort; must not break tool response)
                    if tool_name in AUTO_FORMAT_TOOL_NAMES and tool_run_id:
                        try:
                            format_request = (req_dict.get("format_request") or "").strip() if isinstance(req_dict, dict) else ""
                            resolved = resolve_incremental_format_spec(
                                session_id=str(session_id),
                                turn_id=int(turn_id),
                                source_tool_run_id=tool_run_id,
                                format_request=str(format_request) if format_request else None,
                                format_spec_overrides=None,
                                reset=False,
                                apply_on_latest_presentation_data=False,
                            )
                            spec = resolved["spec"]
                            llm_notes = resolved.get("notes") or []
                            created_mode = resolved.get("created_mode") or ("auto_default" if not format_request else "interpret_request")
                            artifact_row = build_presentation_table_payload_from_tool_run(
                                source_tool_run_id=tool_run_id,
                                spec=spec,
                                created_mode=created_mode,
                                title="Presentation (auto)",
                            )
                            # Attach LLM notes (best-effort) to payload.notes
                            if isinstance(artifact_row.get("payload"), dict):
                                p = artifact_row["payload"]
                                if llm_notes:
                                    notes = p.get("notes") if isinstance(p.get("notes"), list) else []
                                    notes = [str(x) for x in notes if isinstance(x, str)]
                                    notes.extend([str(x) for x in llm_notes if isinstance(x, str) and str(x).strip()])
                                    p["notes"] = notes
                                interp = resolved.get("interpretation")
                                if interp:
                                    fmt = p.get("format") if isinstance(p.get("format"), dict) else {}
                                    fmt["interpretation"] = interp
                                    p["format"] = fmt
                                fmt = p.get("format") if isinstance(p.get("format"), dict) else {}
                                if fmt:
                                    try:
                                        summ = build_format_summary_sv(
                                            spec=spec,
                                            payload_format=fmt,
                                            derived_columns=(fmt.get("derived_columns") if isinstance(fmt.get("derived_columns"), list) else None),
                                            notes=(p.get("notes") if isinstance(p.get("notes"), list) else None),
                                        )
                                        fmt["summary_sv"] = summ.get("summary_sv")
                                        fmt["steps_sv"] = summ.get("steps_sv")
                                        p["format"] = fmt
                                    except Exception:
                                        pass
                            up = upsert_singleton_presentation_artifact(
                                session_id=str(session_id),
                                turn_id=int(turn_id),
                                artifact_row=artifact_row,
                            )
                            artifact_id = up.get("artifact_id")
                            if isinstance(safe_result, dict):
                                meta = safe_result.setdefault("meta", {})
                                if isinstance(meta, dict):
                                    meta.setdefault("presentation_artifact_id", artifact_id)
                        except Exception as e:
                            logger.exception("Auto-format failed for %s tool_run_id=%s: %s", tool_name, tool_run_id, e)

                    if isinstance(safe_result, dict):
                        meta = safe_result.setdefault("meta", {})
                        if isinstance(meta, dict):
                            if tool_run_id:
                                meta.setdefault("tool_run_id", tool_run_id)
                            else:
                                meta.setdefault("tool_run_id", None)

                    return safe_result  # type: ignore[return-value]

                except Exception as exc:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    # Try to log error as a terminal row; don't mask the original exception if logging fails.
                    try:
                        safe_request = jsonable_encoder(req_dict)
                        _try_insert_tool_run(
                            {
                                "session_id": str(session_id),
                                "turn_id": int(turn_id),
                                "tool_name": tool_name,
                                "status": "error",
                                "request_json": safe_request,
                                "error_json": _build_error_json(exc),
                                "duration_ms": duration_ms,
                            },
                            tool_name=tool_name,
                        )
                    except Exception as log_exc:
                        logger.exception("tool_runs insert(error) failed for %s: %s", tool_name, log_exc)
                    raise

        else:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                req_obj = _find_req_obj(args, kwargs)

                req_dict = req_obj.model_dump(mode="json")
                session_id = req_dict.get("session_id")
                turn_id = req_dict.get("turn_id")
                if not session_id or turn_id is None:
                    raise ValueError(f"{tool_name}: session_id and turn_id are required.")

                started = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    duration_ms = int((time.perf_counter() - started) * 1000)

                    safe_result = jsonable_encoder(result) if result is not None else None
                    safe_request = jsonable_encoder(req_dict)
                    tool_run_id = _try_insert_tool_run(
                        {
                            "session_id": str(session_id),
                            "turn_id": int(turn_id),
                            "tool_name": tool_name,
                            "status": "success",
                            "request_json": safe_request,
                            "response_json": safe_result,
                            "duration_ms": duration_ms,
                            "row_count": _infer_row_count(safe_result),
                            "bytes": _estimate_bytes(safe_result),
                        },
                        tool_name=tool_name,
                    )

                    # Auto-create formatted/presentation artifact (best-effort; must not break tool response)
                    if tool_name in AUTO_FORMAT_TOOL_NAMES and tool_run_id:
                        try:
                            format_request = (req_dict.get("format_request") or "").strip() if isinstance(req_dict, dict) else ""
                            resolved = resolve_incremental_format_spec(
                                session_id=str(session_id),
                                turn_id=int(turn_id),
                                source_tool_run_id=tool_run_id,
                                format_request=str(format_request) if format_request else None,
                                format_spec_overrides=None,
                                reset=False,
                                apply_on_latest_presentation_data=False,
                            )
                            spec = resolved["spec"]
                            llm_notes = resolved.get("notes") or []
                            created_mode = resolved.get("created_mode") or ("auto_default" if not format_request else "interpret_request")
                            artifact_row = build_presentation_table_payload_from_tool_run(
                                source_tool_run_id=tool_run_id,
                                spec=spec,
                                created_mode=created_mode,
                                title="Presentation (auto)",
                            )
                            if llm_notes and isinstance(artifact_row.get("payload"), dict):
                                p = artifact_row["payload"]
                                notes = p.get("notes") if isinstance(p.get("notes"), list) else []
                                notes = [str(x) for x in notes if isinstance(x, str)]
                                notes.extend([str(x) for x in llm_notes if isinstance(x, str) and str(x).strip()])
                                p["notes"] = notes
                            up = upsert_singleton_presentation_artifact(
                                session_id=str(session_id),
                                turn_id=int(turn_id),
                                artifact_row=artifact_row,
                            )
                            artifact_id = up.get("artifact_id")
                            if isinstance(safe_result, dict):
                                meta = safe_result.setdefault("meta", {})
                                if isinstance(meta, dict):
                                    meta.setdefault("presentation_artifact_id", artifact_id)
                        except Exception as e:
                            logger.exception("Auto-format failed for %s tool_run_id=%s: %s", tool_name, tool_run_id, e)

                    if isinstance(safe_result, dict):
                        meta = safe_result.setdefault("meta", {})
                        if isinstance(meta, dict):
                            if tool_run_id:
                                meta.setdefault("tool_run_id", tool_run_id)
                            else:
                                meta.setdefault("tool_run_id", None)

                    return safe_result  # type: ignore[return-value]

                except Exception as exc:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    try:
                        safe_request = jsonable_encoder(req_dict)
                        _try_insert_tool_run(
                            {
                                "session_id": str(session_id),
                                "turn_id": int(turn_id),
                                "tool_name": tool_name,
                                "status": "error",
                                "request_json": safe_request,
                                "error_json": _build_error_json(exc),
                                "duration_ms": duration_ms,
                            },
                            tool_name=tool_name,
                        )
                    except Exception as log_exc:
                        logger.exception("tool_runs insert(error) failed for %s: %s", tool_name, log_exc)
                    raise

        # Preserve signature for FastAPI so it doesn't think args/kwargs are query params
        wrapper.__signature__ = sig  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
