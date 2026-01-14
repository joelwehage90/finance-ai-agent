from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class FormatToolRequest(BaseModel):
    # Correlation (where the new artifact should live)
    session_id: str
    turn_id: int

    # Lineage (where the raw data comes from)
    source_tool_run_id: str = Field(..., description="tool_runs.id (UUID) for the raw source payload")

    # v1: explicit structured spec only (no LLM parsing)
    format_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Partial/complete format spec (unit/decimals/top_n/sort/include_totals).",
    )

    # v2 (now): free-text formatting request interpreted by an LLM (OpenAI-compatible)
    format_request: Optional[str] = Field(
        default=None,
        description="Free text like 'i mkr, top 5, sort desc'. Backend will interpret to a format_spec.",
    )

    reset: bool = Field(
        default=False,
        description="If true, start from default_format_spec (ignore the latest presentation_table spec). "
        "Can also be triggered via format_request (LLM).",
    )

    title: Optional[str] = Field(default=None, description="Optional artifact title")
    created_mode: str = Field(default="manual", description="e.g. manual | auto_default | interpret_request")
    parent_artifact_id: Optional[str] = Field(default=None, description="Optional lineage for reformat chains")


