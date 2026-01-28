from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, constr

PeriodYYYYMM = constr(pattern=r"^\d{4}-\d{2}$")


class ReportBuildModule(BaseModel):
    id: str
    type: str


class ModuleConfig(BaseModel):
    type: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class ReportBuildRequest(BaseModel):
    period: PeriodYYYYMM
    modules: List[ReportBuildModule]
    module_configs: Optional[Dict[str, ModuleConfig]] = None


class TableArtifactRef(BaseModel):
    artifact_id: str
    title: str
    presentation_artifact_id: Optional[str] = None
    source_tool_run_id: Optional[str] = None
    source_tool_name: Optional[str] = None
    tool_turn_id: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None


class CommentPlaceholder(BaseModel):
    placeholder_id: str
    status: Literal["empty"]
    text: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class ModuleBuildResult(BaseModel):
    module_id: str
    visible_tables: List[TableArtifactRef]
    supporting_tables: List[TableArtifactRef]
    comment_placeholder: Optional[CommentPlaceholder] = None
    module_status: Literal["ok", "warn", "error"]
    warnings: List[str]


class ReportBuildResponse(BaseModel):
    report_run_id: str
    status: Literal["ok", "error"]
    modules: List[ModuleBuildResult]
    report_spec: Optional[Dict[str, Any]] = None


class PresentationArtifactsRequest(BaseModel):
    artifact_ids: List[str]


class RowGroupingGroupSpec(BaseModel):
    groupId: str
    labels: Dict[str, str]
    memberRowIds: List[str]
    createdAt: str


class RowGroupingSpec(BaseModel):
    enabled: bool = True
    hideMembers: bool = True
    groups: List[RowGroupingGroupSpec] = []


class MaterializeTableViewRequest(BaseModel):
    report_run_id: str
    report_table_id: str
    presentation_artifact_id: str
    module_id: str
    view_spec: Dict[str, Any]
    visibility: Optional[Literal["visible", "supporting"]] = None
    title: Optional[str] = None


class MaterializeTableViewResponse(BaseModel):
    report_table_id: str
    presentation_artifact_id: str
