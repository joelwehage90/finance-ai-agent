from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, constr

PeriodYYYYMM = constr(pattern=r"^\d{4}-\d{2}$")
CompareMode = Literal["month", "ytd"]
FilterValue = Union[str, List[str]]

class VarianceRequest(BaseModel):
    session_id: str
    turn_id: int
    # Free-text formatting request (interpreted by formatting layer/LLM; NOT used by the data query)
    format_request: Optional[str] = None
    compare_mode: CompareMode
    base_period: PeriodYYYYMM
    comp_period: PeriodYYYYMM
    grain: List[str]
    filters: Optional[Dict[str, FilterValue]] = None
    top_n_pos: Optional[int] = Field(default=50, description="null => alla rader")
    top_n_neg: Optional[int] = Field(default=50, description="null => alla rader")
