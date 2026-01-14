from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator

Mode = Literal["hierarchy", "lookup"]

class AccountMappingRequest(BaseModel):
    # Steg 1: spårning (UI/agent)
    session_id: str
    turn_id: int

    mode: Mode = "hierarchy"

    # lookup-fall:
    accounts: Optional[List[int]] = None
    rr_level_1: Optional[str] = None
    rr_level_2: Optional[str] = None
    konto_typ: Optional[str] = None
    search: Optional[str] = Field(default=None, description="Fritext (matchar kontonamn eller nivåer)")

    limit: int = Field(default=200, ge=1, le=2000)

    @model_validator(mode="after")
    def validate_lookup(self):
        if self.mode == "lookup":
            has_search = bool(self.search and self.search.strip())
            if not (self.accounts or self.rr_level_1 or self.rr_level_2 or self.konto_typ or has_search):
                raise ValueError(
                    "Vid mode='lookup' måste minst ett filter anges "
                    "(accounts/rr_level_1/rr_level_2/konto_typ/search)."
                )
        return self
