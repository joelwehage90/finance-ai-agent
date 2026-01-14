from __future__ import annotations

import json
import logging
import os
import re
import base64
import datetime as _dt
from decimal import Decimal
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple
import time

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)


# ----------------------------
# Policy helpers
# ----------------------------

# Must start with a read query. This blocks many dangerous statements by default.
_READ_START_RE = re.compile(r"(?is)^\s*(with\b|select\b)")

# Block write/DDL keywords and other statement types that are not "read-only" in Postgres.
# Note: some of these are already blocked by _READ_START_RE, but we keep a blocklist as a
# belt-and-suspenders safety net.
_READ_BLOCK_RE = re.compile(
    r"(?is)\b("
    r"insert|update|delete|alter|drop|truncate|create|"
    r"lock|copy|vacuum|analyze|explain|set|reset|do|call|grant|revoke"
    r")\b"
)

# SELECT ... FOR UPDATE/SHARE can lock rows and disturb transactional systems.
_FOR_LOCKING_RE = re.compile(
    r"(?is)\bfor\s+("
    r"update|share|no\s+key\s+update|key\s+share"
    r")\b"
)

# If the user explicitly asks for “all rows/no limit”, we should NOT force a default LIMIT.
_NO_LIMIT_RE = re.compile(
    r"(?is)\b("
    r"alla(\s+rader)?|samtliga(\s+rader)?|utan\s+limit|ingen\s+limit|"
    r"no\s+limit|without\s+limit|all\s+rows|entire\s+table|everything"
    r")\b"
)

# “LIMIT” detection is intentionally light-weight.
_HAS_LIMIT_RE = re.compile(r"(?is)\blimit\b")


def _strip_leading_sql_comments(sql: str) -> str:
    """
    Remove leading SQL comments so policy checks can't be bypassed with `--` / `/* */`.
    This is not a full SQL lexer; it's a pragmatic pre-filter for start-of-statement checks.
    """
    s = sql or ""
    while True:
        s2 = s.lstrip()
        if s2.startswith("--"):
            nl = s2.find("\n")
            if nl == -1:
                return ""
            s = s2[nl + 1 :]
            continue
        if s2.startswith("/*"):
            end = s2.find("*/")
            if end == -1:
                return ""
            s = s2[end + 2 :]
            continue
        return s2


def _single_statement_or_none(sql: str) -> Optional[str]:
    """
    Return the single statement SQL if safe, otherwise None.

    We enforce that SQL contains no semicolons except an optional trailing ';'.
    This is intentionally strict (better safe than sorry) and avoids brittle
    regex-based statement splitting.
    """
    s = (sql or "").strip()
    if not s:
        return None

    # Allow exactly one trailing semicolon.
    if ";" in s:
        # Strip trailing semicolons/spaces and ensure nothing else remains.
        stripped = s.rstrip()
        if stripped.endswith(";"):
            body = stripped.rstrip(";").rstrip()
            if ";" in body:
                return None
            return body
        return None

    return s


def _policy_blocks(sql: str) -> Optional[str]:
    """
    Returns a human-readable reason if blocked, else None.
    """
    one = _single_statement_or_none(sql)
    if one is None:
        return "Only single-statement SQL is allowed, and ';' is only permitted as an optional trailing terminator."

    normalized = _strip_leading_sql_comments(one)
    if not _READ_START_RE.search(normalized):
        return "Only read-only queries are allowed (must start with SELECT or WITH)."

    if _READ_BLOCK_RE.search(normalized):
        return "Query contains a blocked keyword (write/DDL or non-read-only statement)."

    # Must not use locking clauses.
    if _FOR_LOCKING_RE.search(normalized):
        return "Row-locking queries are not allowed (FOR UPDATE/SHARE)."
    return None


def _maybe_force_limit(*, sql: str, question: str, top_k: int) -> Tuple[str, Optional[str]]:
    """
    If SQL lacks LIMIT and we should apply a default limit, append `LIMIT top_k`.
    Returns: (sql, warning_or_none)
    """
    s = (sql or "").strip().rstrip(";").strip()
    if top_k <= 0:
        return s, None
    if _HAS_LIMIT_RE.search(s):
        return s, None
    if _NO_LIMIT_RE.search(question or ""):
        return s, None
    return f"{s}\nLIMIT {int(top_k)}", f"Applied default LIMIT {int(top_k)} (top_k)."


# ----------------------------
# LangChain wiring (cached)
# ----------------------------


def _openai_base_url() -> str:
    # Keep the NL→SQL tool decoupled from app/settings.py so it can be imported/used
    # standalone (e.g. in a notebook) without requiring Supabase/API_KEY settings.
    raw = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if not raw:
        raw = "https://api.openai.com/v1"

    base = raw.rstrip("/")

    # Common footgun: setting OPENAI_BASE_URL to https://api.openai.com (missing /v1)
    # which results in requests hitting the wrong path and returning HTML 404.
    if base == "https://api.openai.com":
        return "https://api.openai.com/v1"

    return base


def _openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return api_key


def _openai_model() -> str:
    # Keep consistent with existing formatting tool default.
    return os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


def _statement_timeout_ms() -> int:
    raw = os.environ.get("NL_SQL_STATEMENT_TIMEOUT_MS") or ""
    raw = raw.strip()
    if not raw:
        return 10_000
    try:
        v = int(raw)
        return max(1000, v)  # guardrail: minimum 1s
    except Exception:
        return 10_000


def _max_iterations() -> int:
    raw = os.environ.get("NL_SQL_MAX_ITERATIONS") or ""
    raw = raw.strip()
    if not raw:
        return 12
    try:
        v = int(raw)
        return max(3, min(50, v))
    except Exception:
        return 12


def _db_uri() -> str:
    uri = os.environ.get("DB_URI") or ""
    uri = uri.strip()
    if not uri:
        raise RuntimeError("Missing DB_URI (read-only Postgres connection string).")
    return uri


@lru_cache(maxsize=1)
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=_openai_model(),
        temperature=0,
        api_key=_openai_api_key(),
        base_url=_openai_base_url(),
        # Keep retries low; we already have outer retries in the endpoint tool.
        max_retries=2,
        timeout=60,
    )


@lru_cache(maxsize=1)
def _build_db() -> SQLDatabase:
    """
    NOTE: statement timeout is set at connection level via libpq `options`.
    For production you can also enforce it server-side:
    - role level:   ALTER ROLE <user> SET statement_timeout = '10s';
    - database:     ALTER DATABASE <db> SET statement_timeout = '10s';
    """
    timeout_ms = _statement_timeout_ms()

    # Apply statement_timeout only for Postgres/libpq connections.
    # Other dialects (e.g. sqlite) don't support libpq connect_args like "options".
    uri = _db_uri()
    uri_l = uri.lower().strip()
    is_postgres = uri_l.startswith(
        (
            "postgres://",
            "postgresql://",
            "postgresql+psycopg://",
        )
    )

    return SQLDatabase.from_uri(
        uri,
        engine_args=(
            {
                "connect_args": {
                    # Postgres/libpq option. Works for psycopg v3 as well (passed to libpq).
                    "options": f"-c statement_timeout={int(timeout_ms)}"
                }
            }
            if is_postgres
            else {}
        ),
        view_support=True,
        sample_rows_in_table_info=2,
        max_string_length=2000,
        lazy_table_reflection=True,
    )


@lru_cache(maxsize=1)
def _build_agent_executor_default() -> Any:
    """
    Build a cached AgentExecutor. Note that `top_k` is per-request, so we create
    the agent per request (cheap) but reuse the expensive building blocks (LLM + engine).
    """
    # Intentionally return the toolkit so the endpoint can build a per-request agent.
    db = _build_db()
    llm = _build_llm()
    return SQLDatabaseToolkit(db=db, llm=llm)


class _SafeSQLDatabaseToolkit(SQLDatabaseToolkit):
    """
    Overrides `sql_db_query` with a policy-enforcing wrapper.

    Why: the agent may call `sql_db_query` during its own reasoning. We must ensure
    it cannot execute write/DDL SQL even if prompted/injected.
    """

    def get_tools(self) -> list[Any]:
        tools = list(super().get_tools())

        def _wrap_query_tool(original: Any) -> Any:
            def _safe_query(sql: str) -> Any:
                reason = _policy_blocks(sql)
                if reason:
                    # IMPORTANT: return a tool observation instead of raising.
                    # If we raise, the agent may fail hard and lose the chance to self-repair.
                    return f"POLICY_BLOCK: {reason}"
                return original.invoke(sql)

            return Tool(
                name=original.name,
                description=original.description,
                func=_safe_query,
            )

        out: list[Any] = []
        for t in tools:
            # IMPORTANT: remove sql_db_query entirely so the agent cannot execute.
            # Execution is centralized in the wrapper (run_nl_sql) to avoid double DB calls.
            if getattr(t, "name", None) == "sql_db_query":
                continue
            if getattr(t, "name", None) == "sql_db_query_checker":
                out.append(t)
                continue
            out.append(t)
        return out


def _agent_prefix(*, top_k: int) -> str:
    # Keep it strict: list tables → schema → draft SQL → (optional) checker → return SQL.
    # IMPORTANT: The outer caller controls `top_k` per request. The agent MUST follow it.
    # - top_k > 0 => default LIMIT top_k unless user explicitly asks otherwise
    # - top_k == 0 => do NOT apply any default LIMIT (only use LIMIT if user asks for it)
    if int(top_k) <= 0:
        limit_rule = (
            "- Do not apply any default LIMIT. Only use LIMIT if the user explicitly asks for it.\n"
        )
    else:
        limit_rule = (
            f"- If the user does not explicitly request otherwise, limit results to at most {int(top_k)} rows.\n"
        )
    return "".join(
        [
            "You are an expert Postgres SQL assistant.\n",
            "You must answer the user's question by producing a single read-only SQL query.\n\n",
            "Rules:\n",
            "- Use ONLY the available tools to inspect tables and schema.\n",
            "- Do NOT try to execute SQL during reasoning. You must only generate SQL.\n",
            "- Never use any write/DDL statements (INSERT/UPDATE/DELETE/ALTER/DROP/TRUNCATE/CREATE).\n",
            "- Never use row-locking clauses (FOR UPDATE/SHARE) or other non-read-only statements.\n",
            "- Prefer selecting only the necessary columns (avoid SELECT *).\n",
            limit_rule,
            "- Always double-check your SQL with the query checker tool before returning it.\n\n",
            "Output format:\n",
            "- Return ONLY the SQL query (no markdown, no explanation).\n",
        ]
    )


def _extract_sql_from_agent_result(result: Any) -> Tuple[Optional[str], list[dict[str, Any]]]:
    """
    Best-effort extraction of SQL and a simplified trace from an AgentExecutor result.
    """
    trace: list[dict[str, Any]] = []
    sql: Optional[str] = None

    if isinstance(result, dict):
        # Common keys: "output", "intermediate_steps"
        out = result.get("output")
        if isinstance(out, str) and out.strip():
            sql = out.strip()
            trace.append({"step": "agent_output", "detail": out[:2000]})

        steps = result.get("intermediate_steps")
        if isinstance(steps, list):
            for item in steps:
                # Usually: (AgentAction, observation)
                try:
                    action, observation = item
                except Exception:
                    continue

                tool = getattr(action, "tool", None) if action is not None else None
                tool_input = getattr(action, "tool_input", None) if action is not None else None

                if tool:
                    trace.append({"step": "tool_call", "detail": {"tool": tool, "input": tool_input}})

                # Correctness: Query checker returns the corrected SQL as its *observation*,
                # not as its tool_input. If we want the corrected query, we must read it from
                # the observation.
                if tool == "sql_db_query_checker" and isinstance(observation, str) and observation.strip():
                    sql = observation.strip()

    # Strip accidental code fences.
    if isinstance(sql, str):
        s = sql.strip()
        # Remove ```sql ... ``` wrappers if present
        if s.startswith("```"):
            s = re.sub(r"(?is)^```[a-z0-9_-]*\s*", "", s)
            s = re.sub(r"(?is)\s*```$", "", s).strip()
        sql = s.strip() or None

    return sql, trace


def _execute_select_to_table(*, sql: str) -> Tuple[list[str], list[list[Any]]]:
    """
    Execute SQL and return (columns, rows) in a JSON-friendly shape.
    """
    db = _build_db()
    engine = db._engine  # noqa: SLF001 (LangChain stores the SQLAlchemy engine here)

    def _jsonable(v: Any) -> Any:
        # Keep it small and predictable: only handle the most common Postgres types.
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, Decimal):
            # Avoid float surprises by defaulting to string.
            return str(v)
        if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
            try:
                return v.isoformat()
            except Exception:
                return str(v)
        if isinstance(v, (bytes, bytearray, memoryview)):
            b = bytes(v)
            # base64 ensures the API remains JSON-safe
            return {"__type__": "bytes_b64", "data": base64.b64encode(b).decode("ascii")}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _jsonable(val) for k, val in v.items()}
        return str(v)

    with engine.connect() as conn:
        res = conn.execute(text(sql))
        if not getattr(res, "returns_rows", False):
            return [], []
        cols = list(res.keys())
        rows = [[_jsonable(v) for v in list(r)] for r in res.fetchall()]
        return cols, rows


@dataclass
class NlSqlRunResult:
    sql: Optional[str]
    columns: list[str]
    rows: list[list[Any]]
    warnings: list[str]
    error_code: Optional[str]
    error_message: Optional[str]
    trace: Optional[list[dict[str, Any]]]
    meta: Optional[dict[str, Any]]


def run_nl_sql(*, question: str, top_k: int, include_sql: bool, include_trace: bool) -> NlSqlRunResult:
    """
    NL→SQL runner used by the FastAPI endpoint.

    This function:
    - Uses LangChain SQLDatabaseToolkit + tool-calling agent to generate SQL.
    - Enforces policy (read-only, single statement, optional default LIMIT).
    - Executes SQL via SQLAlchemy and returns tabular data.
    - Retries once or twice on DB errors by asking the agent to repair the query.
    """
    warnings: list[str] = []
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()

    meta: dict[str, Any] = {}

    try:
        # Use the safe toolkit (prevents the agent from executing write/DDL SQL).
        db = _build_db()
        llm = _build_llm()
        toolkit: SQLDatabaseToolkit = _SafeSQLDatabaseToolkit(db=db, llm=llm)
        meta.update(
            {
                "dialect": getattr(db, "dialect", None),
                "model": _openai_model(),
                "statement_timeout_ms": _statement_timeout_ms(),
                "max_iterations": _max_iterations(),
            }
        )
    except Exception as e:
        msg = str(e)
        trace.append({"step": "config_error", "detail": msg})
        return NlSqlRunResult(
            sql=None,
            columns=[],
            rows=[],
            warnings=[],
            error_code="CONFIG_ERROR",
            error_message=msg,
            trace=(trace if include_trace else None),
            meta=None,
        )

    def _invoke_agent(prompt_text: str) -> Tuple[Optional[str], list[dict[str, Any]]]:
        # `create_sql_agent(top_k=...)` is used for prompt guidance inside LangChain.
        # Keep it >= 1 to avoid internal edge cases, but the *policy* is enforced by:
        # - our prefix (which uses the real top_k, including 0)
        # - _maybe_force_limit (which only forces LIMIT when top_k > 0)
        agent_top_k = max(1, int(top_k))
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type="tool-calling",
            top_k=agent_top_k,
            max_iterations=_max_iterations(),
            verbose=False,
            prefix=_agent_prefix(top_k=int(top_k)),
            agent_executor_kwargs={"return_intermediate_steps": True},
        )
        result = agent.invoke({"input": prompt_text})
        return _extract_sql_from_agent_result(result)

    prompt = (question or "").strip()
    if not prompt:
        return NlSqlRunResult(
            sql=None,
            columns=[],
            rows=[],
            warnings=[],
            error_code="INVALID_REQUEST",
            error_message="question is required",
            trace=(trace if include_trace else None),
            meta=None,
        )

    # Attempt 1: generate SQL
    trace.append({"step": "agent_start", "detail": {"attempt": 1}})
    sql, agent_trace = _invoke_agent(prompt)
    trace.extend(agent_trace)

    if not sql:
        return NlSqlRunResult(
            sql=None,
            columns=[],
            rows=[],
            warnings=[],
            error_code="AGENT_ERROR",
            error_message="Agent did not return SQL.",
            trace=(trace if include_trace else None),
            meta=None,
        )

    # Policy gate + limit
    blocked_reason = _policy_blocks(sql)
    if blocked_reason:
        trace.append({"step": "policy_blocked", "detail": {"reason": blocked_reason}})
        return NlSqlRunResult(
            sql=(sql if include_sql else None),
            columns=[],
            rows=[],
            warnings=[],
            error_code="POLICY_BLOCK",
            error_message=blocked_reason,
            trace=(trace if include_trace else None),
            meta=None,
        )

    sql2, limit_warning = _maybe_force_limit(sql=sql, question=prompt, top_k=int(top_k or 0))
    if limit_warning:
        warnings.append(limit_warning)
        trace.append({"step": "policy_limit_applied", "detail": {"top_k": int(top_k), "sql": sql2}})
    sql = sql2

    # Execute (with controlled retries)
    for attempt in [1, 2]:
        try:
            trace.append({"step": "db_execute_start", "detail": {"attempt": attempt}})
            cols, rows = _execute_select_to_table(sql=sql)
            trace.append({"step": "db_execute_ok", "detail": {"row_count": len(rows)}})
            meta["limit_applied"] = any("Applied default LIMIT" in (w or "") for w in warnings)
            meta["execution_ms"] = int((time.perf_counter() - started) * 1000)
            return NlSqlRunResult(
                sql=(sql if include_sql else None),
                columns=cols,
                rows=rows,
                warnings=warnings,
                error_code=None,
                error_message=None,
                trace=(trace if include_trace else None),
                meta=meta,
            )
        except SQLAlchemyError as e:
            msg = str(e)
            trace.append({"step": "db_error", "detail": {"attempt": attempt, "error": msg}})
            if attempt >= 2:
                return NlSqlRunResult(
                    sql=(sql if include_sql else None),
                    columns=[],
                    rows=[],
                    warnings=warnings,
                    error_code="DB_ERROR",
                    error_message=msg,
                    trace=(trace if include_trace else None),
                    meta=meta,
                )

            # Ask the agent to repair the SQL using the error message.
            repair_prompt = (
                f"{prompt}\n\n"
                "The previous SQL caused a database error. Fix the SQL.\n"
                "Return ONLY a single corrected SQL SELECT query.\n\n"
                f"SQL:\n{sql}\n\n"
                f"DB error:\n{msg}\n"
            )
            trace.append({"step": "agent_retry", "detail": {"attempt": attempt + 1}})
            sql_retry, agent_trace2 = _invoke_agent(repair_prompt)
            trace.extend(agent_trace2)
            if sql_retry:
                sql = sql_retry
                # Re-apply policy gates after repair.
                blocked_reason = _policy_blocks(sql)
                if blocked_reason:
                    return NlSqlRunResult(
                        sql=(sql if include_sql else None),
                        columns=[],
                        rows=[],
                        warnings=warnings,
                        error_code="POLICY_BLOCK",
                        error_message=blocked_reason,
                        trace=(trace if include_trace else None),
                        meta=meta,
                    )
                sql, _ = _maybe_force_limit(sql=sql, question=prompt, top_k=int(top_k or 0))
            else:
                return NlSqlRunResult(
                    sql=(sql if include_sql else None),
                    columns=[],
                    rows=[],
                    warnings=warnings,
                    error_code="AGENT_ERROR",
                    error_message="Agent failed to repair SQL after DB error.",
                    trace=(trace if include_trace else None),
                    meta=meta,
                )
        except Exception as e:
            msg = str(e)
            trace.append({"step": "unexpected_error", "detail": {"error": msg}})
            return NlSqlRunResult(
                sql=(sql if include_sql else None),
                columns=[],
                rows=[],
                warnings=warnings,
                error_code="AGENT_ERROR",
                error_message=msg,
                trace=(trace if include_trace else None),
                meta=meta,
            )

    # Unreachable, but keep mypy/linters happy.
    return NlSqlRunResult(
        sql=(sql if include_sql else None),
        columns=[],
        rows=[],
        warnings=warnings,
        error_code="AGENT_ERROR",
        error_message="Unknown error.",
        trace=(trace if include_trace else None),
        meta=meta,
    )


def log_nl_sql_request(
    *,
    session_id: str,
    turn_id: int,
    question: str,
    sql: Optional[str],
    row_count: int,
    error_code: Optional[str],
) -> None:
    """
    Structured JSON log line (in addition to Supabase tool_runs).
    Keep payload small to avoid noisy logs.
    """
    payload = {
        "event": "nl_sql",
        "session_id": session_id,
        "turn_id": int(turn_id),
        "question": (question or "")[:1000],
        "sql": (sql or None),
        "row_count": int(row_count),
        "error_code": error_code,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))

