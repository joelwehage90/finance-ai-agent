from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator

Kind = Literal["business", "metric"]
MatchType = Literal["exact", "search"]

class DefinitionsRequest(BaseModel):
    session_id: str
    turn_id: int
    terms: Optional[List[str]] = Field(default=None, description="Exakta termer/metric_names att slå upp")
    search: Optional[str] = Field(default=None, description="Fritext att söka i definitioner")
    include_business: bool = True
    include_metrics: bool = True
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def validate_input(self):
        if (not self.terms or len(self.terms) == 0) and (not self.search or not self.search.strip()):
            raise ValueError("Du måste ange minst en term (terms) eller en söksträng (search).")
        return self


class DefinitionItem(BaseModel):
    kind: Kind
    name: str
    definition: str
    _match: MatchType

    # Business (valfria fält)
    # NOTE: DB columns can be text, arrays, or json depending on migrations.
    # Keep these permissive to avoid tool/schema mismatches.
    examples: Optional[Any] = None
    example_sql: Optional[Any] = None
    owner: Optional[Any] = None
    updated_at: Optional[Any] = None

    # Metric (valfria fält)
    formula_text: Optional[Any] = None
    scope_filters: Optional[Any] = None
    caveats: Optional[Any] = None
    example_questions: Optional[Any] = None


class DefinitionsResponse(BaseModel):
    meta: Dict[str, Any]
    items: List[DefinitionItem]
