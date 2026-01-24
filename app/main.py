import os
import time
import re
import inspect
import logging
from functools import lru_cache
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .relevance_client import RelevanceAgent, RelevanceClient

from .settings import settings
from .supabase_client import supabase
from .tool_logging import log_tool_call  # <-- viktigt: logging decorator

from .schemas.variance import VarianceRequest
from .schemas.pnl import PnlRequest
from .schemas.format_tool import FormatToolRequest
from .schemas.nl_sql import NlSqlRequest, NlSqlResponse

try:
    from .schemas.definitions import DefinitionsRequest
    from .tools.definitions_tool import definitions_lookup
except Exception:
    DefinitionsRequest = None
    definitions_lookup = None

try:
    from .schemas.account_mapping import AccountMappingRequest
    from .tools.account_mapping_tool import account_mapping_query
except Exception:
    AccountMappingRequest = None
    account_mapping_query = None

from .tools.variance_tool import variance_tables
from .tools.income_statement_tool import pnl_tables
from .tools.nl_sql_tool import log_nl_sql_request, run_nl_sql
from .formatting.format_spec import FormatSpec, default_format_spec, merge_with_default
from .formatting.format_summary import build_format_summary_sv
from .formatting.presentation_table import (
    build_presentation_table_payload_from_tool_run,
    redo_singleton_presentation_artifact,
    undo_singleton_presentation_artifact,
    upsert_singleton_presentation_artifact,
)
from .formatting.spec_pipeline import resolve_incremental_format_spec


logger = logging.getLogger(__name__)

app = FastAPI(title="Finance AI Agent API", version="0.1.0")

# CORS (local dev)
cors_env = os.environ.get("CORS_ORIGINS", "").strip()
if cors_env:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Auth / helpers
# ----------------------------

HISTORY_TOO_LARGE = "This data is too large to store in history"
_META_LINE_RE = re.compile(r"^\[meta\]\s+session_id=.*?\bturn_id=\d+\s*$", re.IGNORECASE)

# Relevance conversation.state normalization buckets (based on your SDK dumps)
RUNNING_STATES = {
    "waiting-for-capacity",
    "queued",
    "pending",
    "running",
    "in_progress",
}
DONE_STATES = {
    "idle",         # IMPORTANT: in your dumps, "idle" is a done-ish terminal state
    "completed",
    "succeeded",
    "success",
    "done",
    "finished",
    "inactive",
}
ERROR_STATES = {
    "failed",
    "error",
    "errored",
    "cancelled",
    "canceled",
    "timeout",
}


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    # Header name becomes "x-api-key" (case-insensitive). Streamlit kan skicka "X-API-Key".
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _latest_agent_text_from_metadata(md_any: Any) -> Optional[Dict[str, Any]]:
    """
    Deterministiskt: försök hämta senaste agent-message från metadata.
    Vi använder conversation.update_date som tidsstämpel (stabilt och lätt att jämföra mot since_ts).
    Returnerar: {"text": str, "update_date": str|None, "update_ts": float}
    
    Tries multiple extraction strategies:
    1. conversation.debug.prompt_completion_output.output.history_items (original)
    2. conversation.debug.prompt_completion_output.output.answer (fallback)
    3. Direct answer field
    """
    md = _dump(md_any)
    if not isinstance(md, dict):
        return None

    conv = md.get("conversation")
    if not isinstance(conv, dict):
        return None

    update_date = conv.get("update_date") or md.get("update_date") or md.get("updateDate")
    update_ts = _parse_iso_dt_to_ts(update_date)

    # Strategy 1: Try history_items (original method)
    hist = (
        conv.get("debug", {})
            .get("prompt_completion_output", {})
            .get("output", {})
            .get("history_items")
    )
    if isinstance(hist, list) and hist:
        # Collect all agent-messages with their timestamps
        candidate_messages: List[Dict[str, Any]] = []
        for item in hist:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "agent-message":
                continue
            t = item.get("text")
            if isinstance(t, str) and t.strip():
                # Try to get timestamp from item, fallback to conversation update_date
                item_ts = _parse_iso_dt_to_ts(item.get("timestamp") or item.get("created_at") or update_date)
                candidate_messages.append({
                    "text": t.strip(),
                    "timestamp": item_ts,
                })
        
        # Sort by timestamp (latest first) and take the most recent one
        if candidate_messages:
            candidate_messages.sort(key=lambda x: x.get("timestamp", 0.0), reverse=True)
            latest_text = candidate_messages[0]["text"]
            return {
                "text": _strip_meta_prefix(latest_text).strip(),
                "update_date": update_date if isinstance(update_date, str) else (str(update_date) if update_date else None),
                "update_ts": float(update_ts),
            }

    # Strategy 2: Try direct answer field
    answer = (
        conv.get("debug", {})
            .get("prompt_completion_output", {})
            .get("output", {})
            .get("answer")
    )
    if isinstance(answer, str) and answer.strip() and not _contains_history_too_large(answer):
        return {
            "text": _strip_meta_prefix(answer).strip(),
            "update_date": update_date if isinstance(update_date, str) else (str(update_date) if update_date else None),
            "update_ts": float(update_ts),
        }

    # Strategy 3: Try top-level answer
    top_answer = md.get("answer") or conv.get("answer")
    if isinstance(top_answer, str) and top_answer.strip() and not _contains_history_too_large(top_answer):
        return {
            "text": _strip_meta_prefix(top_answer).strip(),
            "update_date": update_date if isinstance(update_date, str) else (str(update_date) if update_date else None),
            "update_ts": float(update_ts),
        }

    return None


def _is_new_enough(candidate: Dict[str, Any], *, since_ts: float, epsilon_s: float = 0.5) -> bool:
    """True om candidate.update_ts >= since_ts - epsilon."""
    try:
        cand_ts = float(candidate.get("update_ts", 0.0))
    except Exception:
        cand_ts = 0.0
    return cand_ts >= float(since_ts) - float(epsilon_s)



def _res_data(res: Any) -> List[Dict[str, Any]]:
    # Supabase-py har ibland .data, ibland res.data (property)
    data = getattr(res, "data", None)
    if data is None:
        data = res.data
    return data or []


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    df2 = df.where(pd.notnull(df), None)
    return df2.to_dict(orient="records")


def _contains_history_too_large(x: Any) -> bool:
    """True om payloaden (str/dict/list) innehåller history-too-large-varningen."""
    if x is None:
        return False
    if isinstance(x, bool):
        return False
    if isinstance(x, str):
        return HISTORY_TOO_LARGE in x
    if isinstance(x, dict):
        msg = x.get("message")
        if isinstance(msg, str) and HISTORY_TOO_LARGE in msg:
            return True
        return any(_contains_history_too_large(v) for v in x.values())
    if isinstance(x, list):
        return any(_contains_history_too_large(v) for v in x)
    return False




def _dump(x: Any) -> Any:
    """
    Convert SDK/Pydantic/dataclass objects into JSON-serializable python structures.
    Deterministic and safe: never throws.
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, dict):
        return {str(k): _dump(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_dump(v) for v in x]

    # pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return _dump(x.model_dump(mode="json"))
        except Exception:
            pass

    # pydantic v1
    if hasattr(x, "dict"):
        try:
            return _dump(x.dict())
        except Exception:
            pass

    # dataclass
    try:
        import dataclasses
        if dataclasses.is_dataclass(x):
            return _dump(dataclasses.asdict(x))
    except Exception:
        pass

    # objects with __dict__
    if hasattr(x, "__dict__"):
        try:
            return _dump(vars(x))
        except Exception:
            pass

    return str(x)


def _strip_meta_prefix(text: str) -> str:
    """
    Om modellen ekar första [meta]-raden, ta bort den deterministiskt.
    """
    if not isinstance(text, str):
        return text
    lines = text.splitlines()
    if lines and _META_LINE_RE.match(lines[0].strip()):
        # drop first line, and also drop a single blank line after it if present
        rest = lines[1:]
        if rest and rest[0].strip() == "":
            rest = rest[1:]
        return "\n".join(rest)
    return text


def _parse_iso_dt_to_ts(s: Any) -> float:
    if not s:
        return 0.0
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return 0.0
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _get_conversation_state(md_obj: Any) -> Optional[str]:
    """
    Deterministiskt: state ligger i metadata.conversation.state i din SDK-output.
    Fallbacks finns endast för robusthet.
    """
    d = _dump(md_obj)
    if isinstance(d, dict):
        conv = d.get("conversation")
        if isinstance(conv, dict):
            st = conv.get("state")
            if st is not None:
                return str(st)
        if d.get("state") is not None:
            return str(d.get("state"))
        if d.get("status") is not None:
            return str(d.get("status"))

    conv_obj = getattr(md_obj, "conversation", None)
    if conv_obj is not None:
        st = getattr(conv_obj, "state", None)
        if st is not None:
            return str(st)

    st2 = getattr(md_obj, "state", None) or getattr(md_obj, "status", None)
    return str(st2) if st2 is not None else None


def _state_from_metadata(md_obj: Any) -> Optional[str]:
    """Alias for _get_conversation_state for consistency."""
    return _get_conversation_state(md_obj)


def _normalize_state(state: Optional[str]) -> Optional[str]:
    """
    Normalize state string to one of the known state buckets.
    Returns the state as-is if it matches a known bucket, otherwise returns None.
    """
    if not state:
        return None
    state_lower = str(state).lower().strip()
    all_states = RUNNING_STATES | DONE_STATES | ERROR_STATES
    if state_lower in {s.lower() for s in all_states}:
        return state_lower
    # Return original if it doesn't match known states (for debugging)
    return state




# ----------------------------
# Relevance client/agent
# ----------------------------

def _env_or_setting(env_key: str, setting_attr: str) -> Optional[str]:
    v = os.environ.get(env_key)
    if v:
        return v
    return getattr(settings, setting_attr, None)


@lru_cache(maxsize=1)
def _get_relevance_agent() -> Tuple[RelevanceClient, RelevanceAgent]:
    """
    Deterministiskt: läs config från env (primärt) och settings (fallback).
    """
    api_key = _env_or_setting("RAI_API_KEY", "RAI_API_KEY")
    region = _env_or_setting("RAI_REGION", "RAI_REGION")
    project = _env_or_setting("RAI_PROJECT", "RAI_PROJECT")
    agent_id = _env_or_setting("RAI_AGENT_ID", "RAI_AGENT_ID")

    missing = [k for k, v in [("RAI_API_KEY", api_key), ("RAI_REGION", region), ("RAI_PROJECT", project), ("RAI_AGENT_ID", agent_id)] if not v]
    if missing:
        raise RuntimeError(f"Missing RelevanceAI config: {', '.join(missing)} (set env vars or settings.*)")

    client = RelevanceClient(api_key=str(api_key), region=str(region), project=str(project))
    agent = RelevanceAgent(client=client, agent_id=str(agent_id))
    return client, agent


def _task_id_from_triggered(task_obj: Any) -> Optional[str]:
    # SDK-dokumentation varierar: ibland conversation_id, ibland task_id.
    for attr in ("conversation_id", "task_id", "id"):
        v = getattr(task_obj, attr, None)
        if v:
            return str(v)
    if isinstance(task_obj, dict):
        v = task_obj.get("conversation_id") or task_obj.get("task_id") or task_obj.get("id")
        return str(v) if v else None
    return None


# ----------------------------
# Steps fetching + deterministic "turn response" extraction
# ----------------------------

def _view_task_steps_raw(agent: Any, *, conversation_id: str) -> Any:
    """
    Fetch task steps as *raw JSON* (dict) without SDK pydantic validation.

    Why: the Relevance SDK's `agent.view_task_steps()` constructs `TaskView(**response.json())`,
    which can fail when Relevance returns step items like `content.type = "tool-run"`.
    Using the underlying `_post(..., cast_to=dict)` gives us the raw response safely.
    """
    # We always use the lightweight HTTP client now.
    if hasattr(agent, "view_task_steps_raw"):
        return agent.view_task_steps_raw(conversation_id=conversation_id)
    # Fallback (shouldn't happen): attempt legacy method name.
    if hasattr(agent, "view_task_steps"):
        return agent.view_task_steps(conversation_id=conversation_id)
    raise RuntimeError("Agent does not support view_task_steps.")


def _extract_turn_agent_message_from_steps(
    steps_dict: Any,
    *,
    session_id: str,
    turn_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Deterministiskt: plocka agent-svar för en specifik turn genom att matcha meta-raden
    i user-message och sedan ta sista agent-message efter den user-message:n men innan nästa user-message.

    Returnerar:
      {"text": str, "update_date": str|None, "update_ts": float, "agent_item_id": str|None}
    """
    steps = _dump(steps_dict)
    if not isinstance(steps, dict):
        return None

    results = steps.get("results")
    if not isinstance(results, list) or not results:
        return None

    needle = f"session_id={session_id} turn_id={int(turn_id)}"

    # Normalize and sort in chronological order.
    # Important: API ordering can be newest-first or oldest-first, and timestamp parsing can fail (-> 0.0).
    # We therefore sort primarily by the raw insert_date_ string (ISO-ish, lexicographically sortable),
    # and use the original index as a stable tiebreaker.
    norm: List[Dict[str, Any]] = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        item_type = content.get("type")
        text_val = content.get("text")
        insert_dt = item.get("insert_date_") or item.get("insert_date") or item.get("created_at") or ""
        ts = _parse_iso_dt_to_ts(insert_dt)
        norm.append(
            {
                "_idx": idx,
                "_ts": float(ts),
                "_dt": str(insert_dt) if insert_dt is not None else "",
                "item_id": item.get("item_id") or item.get("id"),
                "insert_date_": insert_dt,
                "type": item_type,
                "text": text_val,
            }
        )

    if not norm:
        return None

    # Chronological sort by insert_date string first; fallback to parsed ts; then index.
    norm.sort(
        key=lambda x: (
            str(x.get("_dt") or ""),
            float(x.get("_ts") or 0.0),
            int(x.get("_idx") or 0),
        )
    )

    # Find the latest matching user-message for this (session_id, turn_id)
    user_pos: Optional[int] = None
    for i, it in enumerate(norm):
        if it.get("type") != "user-message":
            continue
        t = it.get("text")
        if not isinstance(t, str):
            continue
        if needle in t:
            user_pos = i

    if user_pos is None:
        return None

    # Find next user-message boundary (any turn)
    next_user_pos: Optional[int] = None
    for j in range(user_pos + 1, len(norm)):
        if norm[j].get("type") == "user-message":
            next_user_pos = j
            break

    start = user_pos + 1
    end = next_user_pos if next_user_pos is not None else len(norm)

    best: Optional[Dict[str, Any]] = None
    for k in range(start, end):
        it = norm[k]
        if it.get("type") != "agent-message":
            continue
        txt = it.get("text")
        if not isinstance(txt, str) or not txt.strip():
            continue
        if _contains_history_too_large(txt):
            continue
        best = {
            "text": _strip_meta_prefix(txt).strip(),
            "update_date": it.get("insert_date_"),
            "update_ts": float(it.get("_ts") or 0.0),
            "agent_item_id": str(it.get("item_id") or "") or None,
        }

    return best


# ----------------------------
# Agent triggering / continuation
# ----------------------------

class AgentChatRequest(BaseModel):
    session_id: str
    turn_id: int
    message: str
    conversation_id: Optional[str] = None




# ----------------------------
# Health
# ----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Tools (Step 3 logging)
# ----------------------------

@app.post("/tools/variance", dependencies=[Depends(require_api_key)])
@log_tool_call("variance_tool")
def variance_endpoint(req: VarianceRequest) -> Dict[str, Any]:
    # session_id/turn_id are used for tracking/logging, not part of the tool function signature.
    res = variance_tables(**req.model_dump(exclude={"session_id", "turn_id", "format_request"}))
    out: Dict[str, Any] = {}
    for k, v in res.items():
        out[k] = df_to_records(v) if isinstance(v, pd.DataFrame) else v
    return out


@app.post("/tools/income-statement", dependencies=[Depends(require_api_key)])
@log_tool_call("income_statement_tool")
def pnl_endpoint(req: PnlRequest) -> Dict[str, Any]:
    # session_id/turn_id are used for tracking/logging, not part of the tool function signature.
    res = pnl_tables(**req.model_dump(exclude={"session_id", "turn_id", "format_request"}))
    table_df = res["table"]
    return {
        "meta": res.get("meta"),
        "columns": list(table_df.columns),
        "table": df_to_records(table_df),
    }


@app.post("/tools/format", dependencies=[Depends(require_api_key)])
@log_tool_call("format_tool")
def format_endpoint(req: FormatToolRequest) -> Dict[str, Any]:
    """
    Create a presentation_table artifact from an existing tool_run, without re-querying source data.

    v1: default-only + explicit `format_spec` (no LLM parsing).
    """
    resolved = resolve_incremental_format_spec(
        session_id=req.session_id,
        turn_id=int(req.turn_id),
        source_tool_run_id=req.source_tool_run_id,
        format_request=req.format_request,
        format_spec_overrides=req.format_spec,
        reset=bool(req.reset),
        apply_on_latest_presentation_data=True,
    )
    spec: FormatSpec = resolved["spec"]
    notes: List[str] = list(resolved.get("notes") or [])
    effective_source_tool_run_id = str(resolved.get("effective_source_tool_run_id") or req.source_tool_run_id)

    artifact_row = build_presentation_table_payload_from_tool_run(
        source_tool_run_id=effective_source_tool_run_id,
        spec=spec,
        created_mode=str(resolved.get("created_mode") or (req.created_mode or "manual")),
        title=req.title or "Presentation (reformat)",
        artifact_session_id=req.session_id,
        artifact_turn_id=req.turn_id,
    )
    # Attach notes + interpretation into payload
    if isinstance(artifact_row.get("payload"), dict):
        p = artifact_row["payload"]
        if notes:
            existing = p.get("notes") if isinstance(p.get("notes"), list) else []
            existing = [str(x) for x in existing if isinstance(x, str)]
            existing.extend(notes)
            p["notes"] = existing
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
        artifact_row["payload"] = p

    up = upsert_singleton_presentation_artifact(
        session_id=req.session_id,
        turn_id=req.turn_id,
        artifact_row=artifact_row,
    )
    artifact_id = up.get("artifact_id")

    return {
        "artifact_id": artifact_id,
        "artifact_type": "presentation_table",
        "session_id": req.session_id,
        "turn_id": req.turn_id,
        "source_tool_run_id": effective_source_tool_run_id,
        "applied_format_spec": artifact_row.get("format_spec"),
        "changed": bool(up.get("changed")),
        "mode": up.get("mode"),
        "reset": bool(resolved.get("reset")),
    }


@app.post("/tools/nl_sql", dependencies=[Depends(require_api_key)])
@log_tool_call("nl_sql_tool")
def nl_sql_endpoint(req: NlSqlRequest) -> Dict[str, Any]:
    """
    NL→SQL endpoint backed by LangChain SQLDatabaseToolkit + a tool-calling agent.

    This endpoint is designed to be:
    - safe by default (read-only gate + write-block + default LIMIT)
    - observable (tool_runs + structured JSON log line)
    - debuggable (optional simplified trace)
    """
    rr = run_nl_sql(
        question=req.question,
        top_k=req.top_k,
        include_sql=bool(req.include_sql),
        include_trace=bool(req.include_trace),
    )

    # Build a formatter-compatible `table` shape (list[dict]) from columns+rows.
    # Kept alongside the original `columns`+`rows` for backwards compatibility.
    table: List[Dict[str, Any]] = []
    try:
        cols = list(rr.columns or [])
        for row in rr.rows or []:
            if not isinstance(row, list):
                continue
            rec: Dict[str, Any] = {}
            # Only map up to min(len(cols), len(row)) to avoid index errors.
            for i in range(min(len(cols), len(row))):
                key = cols[i]
                if key is None:
                    continue
                rec[str(key)] = row[i]
            table.append(rec)
    except Exception:
        table = []

    resp = NlSqlResponse(
        session_id=req.session_id,
        turn_id=int(req.turn_id),
        sql=rr.sql if req.include_sql else None,
        columns=rr.columns,
        rows=rr.rows,
        table=table,
        row_count=len(rr.rows or []),
        warnings=list(rr.warnings or []),
        error=({"code": rr.error_code, "message": rr.error_message} if rr.error_code else None),
        trace=rr.trace if req.include_trace else None,
        meta=rr.meta,
    )

    log_nl_sql_request(
        session_id=req.session_id,
        turn_id=int(req.turn_id),
        question=req.question,
        sql=resp.sql,
        row_count=int(resp.row_count),
        error_code=(resp.error.code if resp.error else None),
    )

    return resp.model_dump(mode="json")


class FormatUndoRedoRequest(BaseModel):
    session_id: str
    turn_id: int
    source_tool_name: str


@app.post("/tools/format/undo", dependencies=[Depends(require_api_key)])
@log_tool_call("format_undo_tool")
def format_undo_endpoint(req: FormatUndoRedoRequest) -> Dict[str, Any]:
    """
    Undo the active presentation_table for this (session_id, turn_id) without re-querying source data.
    """
    out = undo_singleton_presentation_artifact(
        session_id=req.session_id,
        turn_id=int(req.turn_id),
        source_tool_name=str(req.source_tool_name),
    )
    return {"session_id": req.session_id, "turn_id": int(req.turn_id), **out}


@app.post("/tools/format/redo", dependencies=[Depends(require_api_key)])
@log_tool_call("format_redo_tool")
def format_redo_endpoint(req: FormatUndoRedoRequest) -> Dict[str, Any]:
    """
    Redo the active presentation_table for this (session_id, turn_id) without re-querying source data.
    """
    out = redo_singleton_presentation_artifact(
        session_id=req.session_id,
        turn_id=int(req.turn_id),
        source_tool_name=str(req.source_tool_name),
    )
    return {"session_id": req.session_id, "turn_id": int(req.turn_id), **out}


# Optional tools (only registered if corresponding modules exist)
if DefinitionsRequest is not None and definitions_lookup is not None:

    @app.post("/tools/definitions", dependencies=[Depends(require_api_key)])
    @log_tool_call("definitions_tool")
    def definitions_endpoint(req: DefinitionsRequest) -> Dict[str, Any]:
        # session_id/turn_id are used for tracking/logging, not part of the tool function signature.
        return definitions_lookup(**req.model_dump(exclude={"session_id", "turn_id"}))


if AccountMappingRequest is not None and account_mapping_query is not None:

    @app.post("/tools/account-mapping", dependencies=[Depends(require_api_key)])
    @log_tool_call("account_mapping_tool")
    def account_mapping_endpoint(req: AccountMappingRequest) -> Dict[str, Any]:
        # session_id/turn_id are used for tracking/logging, not part of the tool function signature.
        return account_mapping_query(**req.model_dump(exclude={"session_id", "turn_id"}))


# ----------------------------
# UI retrieval endpoints (för Streamlit artifacts)
# ----------------------------

@app.get("/ui/latest-turn", dependencies=[Depends(require_api_key)])
def latest_turn_endpoint(session_id: str = Query(...)) -> Dict[str, Any]:
    res = (
        supabase
        .table("tool_runs")
        .select("turn_id,created_at")
        .eq("session_id", session_id)
        .order("turn_id", desc=True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = _res_data(res)
    latest_turn_id = rows[0]["turn_id"] if rows else None
    return {"session_id": session_id, "latest_turn_id": latest_turn_id}


@app.get("/ui/tool-runs", dependencies=[Depends(require_api_key)])
def tool_runs_for_turn_endpoint(
    session_id: str = Query(...),
    turn_id: int = Query(...),
    include_request: bool = Query(False),
    include_response: bool = Query(True),
    include_error: bool = Query(True),
    limit: int = Query(200, ge=1, le=2000),
) -> Dict[str, Any]:
    cols = [
        "id",
        "created_at",
        "session_id",
        "turn_id",
        "tool_name",
        "status",
        "duration_ms",
        "row_count",
        "bytes",
    ]
    if include_request:
        cols.append("request_json")
    if include_response:
        cols.append("response_json")
    if include_error:
        cols.append("error_json")

    res = (
        supabase
        .table("tool_runs")
        .select(",".join(cols))
        .eq("session_id", session_id)
        .eq("turn_id", turn_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    rows = _res_data(res)

    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "count": len(rows),
        "runs": rows,
    }


@app.get("/ui/artifacts", dependencies=[Depends(require_api_key)])
def artifacts_for_turn_endpoint(
    session_id: str = Query(...),
    turn_id: int = Query(...),
    artifact_type: Optional[str] = Query(None, description="Optional filter, e.g. presentation_table"),
    include_payload: bool = Query(True),
    include_format_spec: bool = Query(False),
    limit: int = Query(200, ge=1, le=2000),
) -> Dict[str, Any]:
    """
    Fetch formatted/presentation artifacts (stored in public.artifacts) for a given (session_id, turn_id).
    This is separate from raw tool outputs in tool_runs.
    """
    cols = [
        "id",
        "created_at",
        "updated_at",
        "session_id",
        "turn_id",
        "artifact_type",
        "title",
        "created_mode",
        "source_tool_run_id",
        "source_tool_name",
        "parent_artifact_id",
        "row_count",
        "bytes",
    ]
    if include_format_spec:
        cols.append("format_spec")
    if include_payload:
        cols.append("payload")

    q = (
        supabase
        .table("artifacts")
        .select(",".join(cols))
        .eq("session_id", session_id)
        .eq("turn_id", turn_id)
        .order("updated_at", desc=False)
        .order("created_at", desc=False)
        .limit(limit)
    )
    if artifact_type:
        q = q.eq("artifact_type", artifact_type)

    res = q.execute()
    rows = _res_data(res)
    resolved_turn_id: Optional[int] = int(turn_id)

    # Fallback: if no artifacts for this turn, return latest artifacts for the session.
    if not rows:
        q2 = (
            supabase
            .table("artifacts")
            .select(",".join(cols))
            .eq("session_id", session_id)
            .order("turn_id", desc=True)
            .order("updated_at", desc=True)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if artifact_type:
            q2 = q2.eq("artifact_type", artifact_type)
        res2 = q2.execute()
        rows = _res_data(res2)
        resolved_turn_id = int(rows[0]["turn_id"]) if rows else None

    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "requested_turn_id": int(turn_id),
        "resolved_turn_id": resolved_turn_id,
        "count": len(rows),
        "artifacts": rows,
    }


# ----------------------------
# Agent proxy endpoints (RelevanceAI via backend)
# ----------------------------

@app.post("/agent/chat", dependencies=[Depends(require_api_key)])
def agent_chat(req: AgentChatRequest) -> Dict[str, Any]:
    """
    Starta ny task (trigger_task) eller fortsätt i befintlig task (schedule_action_in_task).
    Deterministiskt: returnera aldrig ett svar som inte är "nyare" än detta anrop (since_ts).
    """
    try:
        client, agent = _get_relevance_agent()
    except Exception as e:
        logger.exception("Failed to get RelevanceAI agent")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RelevanceAI agent: {str(e)}"
        )

    # Skicka IDs till agenten via meta-rad
    msg = f"[meta] session_id={req.session_id} turn_id={req.turn_id}\n\n{req.message}"

    # Watermark: används för att avgöra att svaret är nytt för just detta meddelande
    since_ts = time.time()

    conversation_id: Optional[str] = None

    # 1) Fortsätt i samma task om möjligt
    if req.conversation_id:
        try:
            fn = getattr(agent, "schedule_action_in_task", None)
            if callable(fn):
                # According to SDK docs: schedule_action_in_task(conversation_id, message, minutes_until_schedule=0)
                kwargs: Dict[str, Any] = {
                    "conversation_id": req.conversation_id,
                    "message": msg,
                    "minutes_until_schedule": 0  # Immediate execution
                }

                # Some SDK versions may require agent_id parameter
                try:
                    sig = inspect.signature(fn)
                    if "agent_id" in sig.parameters:
                        agent_id = _env_or_setting("RAI_AGENT_ID", "RAI_AGENT_ID")
                        if agent_id:
                            kwargs["agent_id"] = agent_id
                except Exception:
                    pass

                fn(**kwargs)
                conversation_id = req.conversation_id
        except Exception as e:
            logger.warning(f"Failed to schedule action in existing task: {e}, starting new task instead")

    # 2) Annars: starta en ny task
    if not conversation_id:
        try:
            triggered = agent.trigger_task(message=msg)
            conversation_id = _task_id_from_triggered(triggered)
        except Exception as e:
            logger.exception("Failed to trigger new task")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to trigger RelevanceAI task: {str(e)}"
            )

    if not conversation_id:
        return {"status": "error", "assistant_message": "No conversation_id returned."}

    # Lång-polla kort för bra UX.
    # Viktigt: vi matchar svar per turn via view_task_steps + meta-rad (session_id/turn_id),
    # inte via "latest answer", som ofta blir en turn efter.
    deadline = time.time() + 25

    while time.time() < deadline:
        # 1) Try extract the correct turn's agent message from steps
        try:
            steps_raw = _view_task_steps_raw(agent, conversation_id=conversation_id)
            turn_msg = _extract_turn_agent_message_from_steps(
                steps_raw, session_id=req.session_id, turn_id=req.turn_id
            )
            if turn_msg and turn_msg.get("text"):
                return {
                    "status": "completed",
                    "conversation_id": conversation_id,
                    "assistant_message": turn_msg["text"],
                    "since_ts": since_ts,
                    "message_ts": turn_msg.get("update_ts"),
                }
        except Exception as e:
            logger.debug("agent_chat: view_task_steps failed: %s", e)

        # 2) Fallback to metadata state only (to detect failures)
        try:
            md = client.tasks.get_metadata(conversation_id=conversation_id)
            state_norm = _normalize_state(_state_from_metadata(md))
            if state_norm in ERROR_STATES:
                return {
                    "status": "error",
                    "conversation_id": conversation_id,
                    "assistant_message": "Task failed.",
                    "since_ts": since_ts,
                }
        except Exception as e:
            logger.debug("agent_chat: get_metadata failed: %s", e)

        time.sleep(0.8)

    # Inte klar inom 25s => Streamlit fortsätter via /agent/poll
    return {"status": "running", "conversation_id": conversation_id, "since_ts": since_ts}


@app.get("/agent/debug-metadata", dependencies=[Depends(require_api_key)])
def debug_metadata(
    conversation_id: str = Query(...),
    include_full_metadata: bool = Query(False, description="If true, include full raw metadata dump (can be large)."),
) -> Dict[str, Any]:
    """
    Debug endpoint to inspect raw metadata structure.
    Useful for troubleshooting response extraction issues.
    """
    try:
        client, agent = _get_relevance_agent()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")

    try:
        md = client.tasks.get_metadata(conversation_id=conversation_id)
        md_dict = _dump(md)
        
        # Try to extract response using all methods
        cand = _latest_agent_text_from_metadata(md)
        state = _state_from_metadata(md)
        
        # Try using view_task_steps as alternative (compact summary only)
        steps_info: Dict[str, Any]
        try:
            steps_raw = _view_task_steps_raw(agent, conversation_id=conversation_id)
            steps_dict = _dump(steps_raw)

            results = steps_dict.get("results", [])
            if not isinstance(results, list):
                results = []

            # Build a small tail preview of steps (last 8 items, trimmed text)
            tail = results[-8:] if len(results) > 8 else results
            tail_preview: List[Dict[str, Any]] = []
            for item in tail:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, dict):
                    continue
                t = content.get("text")
                t_str = t if isinstance(t, str) else ""
                t_str = _strip_meta_prefix(t_str).strip()
                if len(t_str) > 200:
                    t_str = t_str[:200] + "…"
                tail_preview.append(
                    {
                        "type": content.get("type"),
                        "insert_date_": item.get("insert_date_") or item.get("insert_date") or item.get("created_at"),
                        "item_id": item.get("item_id") or item.get("id"),
                        "text": t_str,
                    }
                )

            steps_info = {
                "has_steps": True,
                "results_count": len(results),
                "tail_preview": tail_preview,
            }
        except Exception as e:
            steps_info = {"error": str(e)}
        
        # Try get_task_output_preview (usually small)
        preview_info: Any
        try:
            preview = agent.get_task_output_preview(conversation_id=conversation_id)
            preview_info = _dump(preview)
        except Exception as e:
            preview_info = {"error": str(e), "method_available": hasattr(agent, "get_task_output_preview")}
        
        out: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "state": state,
            "extracted_candidate": cand,
            "metadata_structure": {
                "top_level_keys": list(md_dict.keys()) if isinstance(md_dict, dict) else [],
                "has_conversation": "conversation" in md_dict if isinstance(md_dict, dict) else False,
                "conversation_keys": list(md_dict.get("conversation", {}).keys()) if isinstance(md_dict, dict) and isinstance(md_dict.get("conversation"), dict) else [],
            },
            "steps_info": steps_info,
            "preview_info": preview_info,
        }

        if include_full_metadata:
            out["full_metadata"] = md_dict

        return out
    except Exception as e:
        logger.exception(f"Debug metadata failed for {conversation_id}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@app.get("/agent/poll", dependencies=[Depends(require_api_key)])
def agent_poll(
    conversation_id: str,
    since_ts: float = Query(..., description="Epoch-sekunder från /agent/chat. Används för att deterministiskt hitta *nytt* svar."),
    session_id: str = Query(..., description="Session id (used to match the correct turn in steps)."),
    turn_id: int = Query(..., description="Turn id (used to match the correct turn in steps)."),
    include_meta: bool = Query(False),
) -> Dict[str, Any]:
    try:
        client, agent = _get_relevance_agent()
    except Exception as e:
        logger.exception("Failed to get RelevanceAI agent in poll")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RelevanceAI agent: {str(e)}"
        )

    # Turn-aware extraction (Streamlit client). If the turn answer isn't found yet, keep polling.
    try:
        steps_raw = _view_task_steps_raw(agent, conversation_id=conversation_id)
        turn_msg = _extract_turn_agent_message_from_steps(
            steps_raw, session_id=str(session_id), turn_id=int(turn_id)
        )
        if turn_msg and turn_msg.get("text"):
            return {
                "status": "completed",
                "conversation_id": conversation_id,
                "since_ts": since_ts,
                "assistant_message": turn_msg["text"],
                "message_ts": turn_msg.get("update_ts"),
            }
    except Exception as e:
        logger.debug("agent_poll: view_task_steps failed: %s", e)

    # Not found yet: return running (or error if task is in an error state)
    try:
        md = client.tasks.get_metadata(conversation_id=conversation_id)
        state_norm = _normalize_state(_state_from_metadata(md))
        if state_norm in ERROR_STATES:
            return {"status": "error", "conversation_id": conversation_id, "since_ts": since_ts}
    except Exception:
        md = None

    out: Dict[str, Any] = {"conversation_id": conversation_id, "since_ts": since_ts, "status": "running"}
    if include_meta and md is not None:
        out["meta"] = _dump(md)
    return out
