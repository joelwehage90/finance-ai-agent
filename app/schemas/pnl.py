from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, constr

PeriodYYYYMM = constr(pattern=r"^\d{4}-\d{2}$")
CompareMode = Literal["month", "ytd"]
FilterValue = Union[str, List[str]]

class PnlRequest(BaseModel):
    session_id: str
    turn_id: int
    # Free-text formatting request (interpreted by LLM in formatting layer; NOT used by the data query)
    format_request: Optional[str] = None
    compare_mode: CompareMode
    periods: List[PeriodYYYYMM]
    rows: List[str]
    filters: Optional[Dict[str, FilterValue]] = None
    include_total: bool = True
