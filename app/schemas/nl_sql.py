from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NlSqlRequest(BaseModel):
    """
    Request contract for NL→SQL.

    Notes:
    - `top_k` is used as a default LIMIT if the generated SQL does not include an explicit LIMIT.
    - `session_id` + `turn_id` are correlation keys used across the system (UI ↔ tools ↔ tool_runs).
    """

    question: str = Field(..., min_length=1, description="Natural language question to be answered via SQL.")
    session_id: str = Field(
        ...,
        description="Correlation id for a conversation/session (used for tool_runs + UI retrieval).",
    )
    turn_id: int = Field(
        ...,
        ge=0,
        description="Turn index within the session (integer; used for ordering + correlation).",
    )

    top_k: int = Field(default=50, ge=0, description="Default LIMIT if SQL has no explicit LIMIT.")
    include_sql: bool = Field(default=True, description="If true, return the executed SQL when available.")
    include_trace: bool = Field(default=False, description="If true, return a simplified tool/agent trace.")
    format_request: Optional[str] = Field(
        default=None,
        description="Optional formatting request for the auto-created presentation_table (e.g. 'i mkr, top 5').",
    )


class NlSqlError(BaseModel):
    code: str
    message: str


class NlSqlTraceStep(BaseModel):
    step: str
    detail: Any


class NlSqlMeta(BaseModel):
    """
    Small, stable metadata surface for debugging/ops.
    """

    dialect: Optional[str] = Field(default=None, description="Database dialect reported by SQLDatabase (e.g. 'postgresql').")
    model: Optional[str] = Field(default=None, description="LLM model used for NL→SQL.")
    statement_timeout_ms: Optional[int] = Field(default=None, description="Configured statement timeout (ms), if applied.")
    max_iterations: Optional[int] = Field(default=None, description="Agent max_iterations (upper bound for self-repair loops).")
    limit_applied: bool = Field(default=False, description="True if the wrapper applied a default LIMIT (top_k).")
    execution_ms: Optional[int] = Field(default=None, description="Total NL→SQL execution time in milliseconds.")


class NlSqlResponse(BaseModel):
    session_id: str = Field(
        ...,
        description="Correlation id for a conversation/session (used for tool_runs + UI retrieval).",
    )
    turn_id: int = Field(
        ...,
        ge=0,
        description="Turn index within the session (integer; used for ordering + correlation).",
    )

    sql: Optional[str] = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    # Formatter-compatible shape (derived from columns+rows). Kept for UI/artifacts.
    table: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0

    warnings: list[str] = Field(default_factory=list)
    error: Optional[NlSqlError] = None
    trace: Optional[list[NlSqlTraceStep]] = None
    meta: Optional[NlSqlMeta] = None

