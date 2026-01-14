from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator, model_validator


Unit = Literal["sek", "tsek", "msek"]
SortDir = Literal["asc", "desc"]
DeriveOp = Literal["sub", "add", "mul", "div", "ratio", "pct_change", "abs", "neg", "sum"]
FilterOp = Literal["eq", "neq", "in", "not_in", "contains", "gt", "gte", "lt", "lte"]
BoolOp = Literal["and", "or"]


_UNIT_ALIASES: Dict[str, Unit] = {
    # canonical
    "sek": "sek",
    "tsek": "tsek",
    "msek": "msek",
    # common aliases
    "kr": "sek",
    "tkr": "tsek",
    "mkr": "msek",
    # punctuation / case variants (normalized before lookup)
}


class SortKey(BaseModel):
    """
    Sort instruction.

    Note: `col` is intentionally optional to support requests like "sort desc" or "top 5"
    where the user didn't name the value column. We resolve `col` deterministically later.
    """

    col: Optional[str] = Field(default=None, description="Column name to sort by. If omitted, resolver picks default.")
    dir: SortDir = Field(default="desc")
    order: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional categorical ordering for string columns. If provided, rows are grouped/sorted by the first "
            "matching token in this list (case-insensitive contains-match). Non-matching values come after the list."
        ),
    )

    @field_validator("col")
    @classmethod
    def _clean_col(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("order", mode="before")
    @classmethod
    def _clean_order(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, list):
            return None
        cleaned: List[str] = []
        for x in v:
            xs = str(x).strip()
            if not xs:
                continue
            cleaned.append(xs)
            if len(cleaned) >= 20:
                break
        return cleaned or None


class DeriveSpec(BaseModel):
    """
    v1 derived column spec.

    Binary ops (require a+b):
    - sub:        out = a - b
    - add:        out = a + b
    - mul:        out = a * b
    - div/ratio:  out = a / b (safe div0 -> null)
    - pct_change: out = (a - b) / b  (fraction; safe div0 -> null)

    Unary ops (require only a):
    - abs: out = abs(a)
    - neg: out = -a

    Multi-input ops:
    - sum: out = sum(inputs)

    Column references must be existing numeric columns in the current table.
    """

    name: str = Field(..., min_length=1, max_length=64)
    op: DeriveOp
    a: Optional[str] = Field(default=None, min_length=1, max_length=128)
    b: Optional[str] = Field(default=None, min_length=1, max_length=128)
    inputs: Optional[List[str]] = Field(default=None, description="For op='sum': list of column names to sum.")

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        return str(v).strip()

    @field_validator("a", "b", mode="before")
    @classmethod
    def _clean_opt_str(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @model_validator(mode="after")
    def _validate_operands(self) -> "DeriveSpec":
        unary = {"abs", "neg"}
        binary = {"sub", "add", "mul", "div", "ratio", "pct_change"}
        multi = {"sum"}

        if self.op in unary:
            if not self.a:
                raise ValueError(f"derive op {self.op} requires 'a'")
            self.b = None
            self.inputs = None
            return self

        if self.op in binary:
            if not self.a or not self.b:
                raise ValueError(f"derive op {self.op} requires 'a' and 'b'")
            self.inputs = None
            return self

        if self.op in multi:
            if not self.inputs or len(self.inputs) < 2:
                raise ValueError("derive op sum requires inputs (>=2)")
            # sum ignores a/b
            self.a = None
            self.b = None
            # normalize/strip inputs
            cleaned: List[str] = []
            for x in self.inputs:
                xs = str(x).strip()
                if xs:
                    cleaned.append(xs)
            self.inputs = cleaned[:10] or None
            if not self.inputs or len(self.inputs) < 2:
                raise ValueError("derive op sum requires inputs (>=2)")
            return self

        return self


class FilterRule(BaseModel):
    """
    Declarative row filter.

    - col: column name in the current table (before rename)
    - op: operation (string or numeric)
    - value: scalar or list (for in/not_in)
    - id: optional stable id for incremental merge across turns
    """

    id: Optional[str] = Field(default=None, max_length=80)
    col: str = Field(..., min_length=1, max_length=128)
    op: FilterOp
    value: Optional[Union[str, int, float, bool, List[Union[str, int, float, bool]]]] = None

    @field_validator("id", mode="before")
    @classmethod
    def _clean_id(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("col", mode="before")
    @classmethod
    def _clean_col(cls, v: Any) -> str:
        return str(v).strip()


class FilterGroup(BaseModel):
    """
    Group of filters combined with a boolean operator (and/or).

    Semantics in v1:
    - Each group is evaluated on a row to produce True/False.
    - All groups are AND'ed together at the top level (i.e. row must satisfy every group).

    This enables expressions like:
      (A OR B) AND (C OR D)
    """

    id: Optional[str] = Field(default=None, max_length=80)
    op: BoolOp = Field(default="or")
    rules: List[FilterRule] = Field(default_factory=list, min_length=1, max_length=20)

    @field_validator("id", mode="before")
    @classmethod
    def _clean_id(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @model_validator(mode="after")
    def _normalize_rules(self) -> "FilterGroup":
        # cap to keep v1 sane
        if self.rules and len(self.rules) > 20:
            self.rules = self.rules[:20]
        return self


FilterExpr = Dict[str, Any]


def _count_filter_expr_nodes(expr: Any, *, limit: int = 200) -> int:
    """
    Best-effort node counter to avoid pathological expressions.
    Supports shapes:
      - {"op": "and"|"or", "args":[...]}
      - {"op": "not", "arg": {...}}
      - {"col":..., "op":..., "value":...} (leaf)
      - {"rule": {...}} (leaf wrapper)
    """
    if expr is None:
        return 0
    if not isinstance(expr, dict):
        return 0
    n = 1
    if n > limit:
        return n
    op = str(expr.get("op") or "").lower()
    if op in {"and", "or"} and isinstance(expr.get("args"), list):
        for a in expr.get("args") or []:
            n += _count_filter_expr_nodes(a, limit=limit)
            if n > limit:
                return n
    elif op == "not" and isinstance(expr.get("arg"), dict):
        n += _count_filter_expr_nodes(expr.get("arg"), limit=limit)
    elif "rule" in expr and isinstance(expr.get("rule"), dict):
        n += _count_filter_expr_nodes(expr.get("rule"), limit=limit)
    return n


class FormatSpec(BaseModel):
    """
    Minimal v1 formatting spec (default-only first).

    We keep it UI/tool-agnostic. Validation here ensures:
    - canonical units
    - bounded decimals/top_n
    - normalized sort structure
    """

    unit: Unit = Field(default="sek")
    decimals: int = Field(default=0, ge=0, le=3)
    top_n: Optional[int] = Field(default=None, ge=1, le=100)
    sort: Optional[List[SortKey]] = Field(default=None, description="Optional list of sort keys (priority order).")
    include_totals: Optional[bool] = Field(default=None)
    column_decimals: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional per-column decimals override (0..3). Keys are column names as they exist before rename.",
    )
    filters: Optional[List[FilterRule]] = Field(
        default=None,
        description="Optional row filters applied after derive and before sort/top_n.",
    )
    filter_groups: Optional[List[FilterGroup]] = Field(
        default=None,
        description="Optional filter groups to support OR logic. All groups are AND'ed at top level.",
    )
    filter_expr: Optional[FilterExpr] = Field(
        default=None,
        description="Optional nested boolean filter expression tree (v2). If set, it takes precedence over filters/filter_groups.",
    )
    rename_columns: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional mapping {old_col: new_col}. Applied at presentation layer (payload) after formatting.",
    )
    derive: Optional[List[DeriveSpec]] = Field(
        default=None,
        description="Optional derived columns computed from the existing table (no re-query).",
    )

    @field_validator("unit", mode="before")
    @classmethod
    def _normalize_unit(cls, v: Any) -> Unit:
        if v is None:
            return "sek"
        s = str(v).strip().lower()
        s = s.replace(".", "").replace(" ", "")
        unit = _UNIT_ALIASES.get(s)
        if unit:
            return unit
        raise ValueError("unit must be one of: sek, tsek, msek (aliases: kr, tkr, mkr)")

    @model_validator(mode="after")
    def _normalize_sort_list(self) -> "FormatSpec":
        if self.sort is not None and len(self.sort) == 0:
            self.sort = None
        # Cap sort keys to keep things sane (v1)
        if self.sort is not None and len(self.sort) > 3:
            self.sort = self.sort[:3]

        # Normalize rename map (strip, drop empties, cap size)
        if self.rename_columns is not None:
            cleaned: Dict[str, str] = {}
            for k, v in list(self.rename_columns.items()):
                k2 = str(k).strip()
                v2 = str(v).strip()
                if not k2 or not v2:
                    continue
                cleaned[k2] = v2
                if len(cleaned) >= 20:
                    break
            self.rename_columns = cleaned or None

        # Normalize column_decimals (strip, validate 0..3, cap size)
        if self.column_decimals is not None:
            cleaned_cd: Dict[str, int] = {}
            for k, v in list(self.column_decimals.items()):
                k2 = str(k).strip()
                if not k2:
                    continue
                try:
                    d = int(v)
                except Exception:
                    continue
                if 0 <= d <= 3:
                    cleaned_cd[k2] = d
                if len(cleaned_cd) >= 20:
                    break
            self.column_decimals = cleaned_cd or None

        # Normalize filters (drop empties, cap count)
        if self.filters is not None:
            if len(self.filters) == 0:
                self.filters = None
            else:
                self.filters = self.filters[:10]

        # Normalize filter_groups (drop empties, cap count)
        if self.filter_groups is not None:
            if len(self.filter_groups) == 0:
                self.filter_groups = None
            else:
                self.filter_groups = self.filter_groups[:5]

        # Bound filter_expr size (v2)
        if self.filter_expr is not None:
            try:
                n_nodes = _count_filter_expr_nodes(self.filter_expr, limit=200)
                if n_nodes > 200:
                    raise ValueError("filter_expr too large (max 200 nodes)")
            except Exception as e:
                raise ValueError(f"Invalid filter_expr: {e}")

        # Normalize derive list (drop empties, cap count)
        if self.derive is not None:
            if len(self.derive) == 0:
                self.derive = None
            else:
                self.derive = self.derive[:5]
        return self


def default_format_spec() -> FormatSpec:
    """
    v1 default-only:
    - decimals = 0 (thousand separators are handled at presentation/render time)
    - unit defaults to SEK
    - default sorting: desc by the "rightmost" column (resolved deterministically)
    """

    # Note: sort.col is intentionally omitted; resolver will pick the default column (rightmost).
    return FormatSpec(
        unit="sek",
        decimals=0,
        top_n=None,
        sort=[SortKey(col=None, dir="desc")],
        include_totals=True,
    )


def resolve_missing_sort_columns(spec: FormatSpec, *, columns: Sequence[str]) -> FormatSpec:
    """
    Deterministic rule for missing sort.col:
    - Use the rightmost column (columns[-1]) as the default value column.

    This is tool-agnostic and matches tables where "latest period" is rightmost.
    """

    cols = [str(c) for c in columns if c is not None]
    if not cols or not spec.sort:
        return spec

    default_col = cols[-1]
    changed = False
    out_sort: List[SortKey] = []
    for sk in spec.sort:
        # Defensive: older/buggy callers may pass raw dicts instead of SortKey objects.
        sk2: Optional[SortKey]
        if isinstance(sk, SortKey):
            sk2 = sk
        else:
            try:
                sk2 = SortKey.model_validate(sk)
                changed = True
            except Exception:
                continue

        if sk2.col is None:
            out_sort.append(SortKey(col=default_col, dir=sk2.dir))
            changed = True
        else:
            out_sort.append(sk2)

    if not changed:
        return spec
    return spec.model_copy(update={"sort": out_sort})


def merge_with_default(spec: Optional[FormatSpec]) -> FormatSpec:
    """
    Merge a possibly-partial spec with defaults.
    (In v1, we'll mostly use defaults only, but this will be handy in v2.)
    """

    # IMPORTANT:
    # - `model_copy(update=...)` does NOT validate/coerce types.
    # - We therefore re-validate with FormatSpec.model_validate on the merged dict,
    #   so nested models like SortKey stay typed (not plain dicts).
    base = default_format_spec()
    if spec is None:
        return base

    merged = base.model_dump(mode="json")
    merged.update(spec.model_dump(mode="json", exclude_unset=True))
    return FormatSpec.model_validate(merged)


