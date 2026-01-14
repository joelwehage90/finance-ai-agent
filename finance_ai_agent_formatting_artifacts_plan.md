# Finance AI Agent — Formatting Artifacts (format_tool) Plan & Requirements

This document describes the architectural goal and concrete implementation requirements for the “formatting artifacts” layer in my Finance AI Agent project. It is intended to be pasted into Codex (Cursor) to give full context before implementing code changes.

---

## 1) Big Picture

I am building a Finance AI Agent with this stack:

- **FastAPI backend** (Python) exposing “tools” as HTTP endpoints
- **Supabase Postgres** as data store and logging store
- **RelevanceAI agent** orchestrating tool calls (the agent calls my FastAPI tools)
- **Streamlit UI** that:
  - sends user messages to an `/agent/chat` endpoint in my FastAPI backend
  - displays artifacts created during the session (tables/charts, etc.)
- Deployed versions may run on Render, but local dev is standard.

Current system already has:

- `session_id: str` and `turn_id: int` tracked end-to-end
- A `tool_runs` table in Supabase that logs every tool call (request/response) and includes a primary key `id` (UUID).
- Tool logging/decorator layer that ensures each tool run is logged and that the tool response includes `meta.tool_run_id`.
- Artifacts rendering in UI already works in principle (the UI can pull artifacts from Supabase), but we now want to improve how *formatted/presented* artifacts are generated.

---

## 2) Problem & Objective

### Current State
When a user asks something like:
- “Visa resultatet 2025-02”
- “Visa i mkr, top 5”

the agent can call a data tool (e.g. `income_statement`) that returns a “raw” table. The UI can show that raw table as an artifact.

### Desired State
I want to support **a second artifact**: a **formatted / presentation table**, similar to how the LLM would normally present a table in chat:
- unit conversion (e.g. SEK → Mkr)
- sorting (e.g. descending by amount)
- top-N (e.g. top 5)
- optional totals behavior
- basic derived columns in v1 (see compute/derive below)

The formatted artifact should be created in a **generic, reusable, deterministic** way, so it works both for Streamlit and for a future UI (Lovable). The formatted artifact should be stored in Supabase and displayed in UI as an artifact, separate from the chat text.

**Important clarification (v1)**
- “Raw artifacts” today are effectively the logged tool outputs in `tool_runs.response_json`.
- “Formatted / presentation artifacts” will be stored separately in the new `artifacts` table.
- UI should be able to show both:
  - raw tool outputs (from `tool_runs`)
  - presentation tables (from `artifacts`)

---

## 3) Key Design Decisions

### 3.1 Separate `format_tool`
We will implement formatting as a dedicated “tool” / module (not embedded into every data tool). This tool can be:
- **triggered automatically** after data tools run, and also
- **called explicitly** when a user wants to reformat without re-fetching data.

### 3.2 Deterministic pipeline
The format workflow is:

1. **Input:** a user formatting request (optional) such as “i mkr, top 5, sort desc”
2. **LLM step (only when needed):** convert free text → a structured `format_spec`
3. **Deterministic resolver (always):**
   - load source data (from `tool_runs.response_json`)
   - apply filter/sort/topN/unit/rounding per `format_spec`
   - optionally apply allowed derived columns (v1 compute)
   - produce a stable `payload` JSON suitable for any UI
4. **Store:** insert artifact row into Supabase `artifacts` table with lineage.

### 3.3 Auto-trigger behavior (backend-owned)
Auto-trigger will live in the **FastAPI backend** (not only inside RelevanceAI workflows), to keep it generic for future UIs:
- RelevanceAI tool calls my FastAPI endpoints
- backend logs tool run in Supabase
- backend triggers format artifact creation after certain tool endpoints return

**Implementation choice (v1)**
- Put the auto-trigger logic directly in the tool endpoint (starting with `/tools/income-statement`) after the tool run has been logged and we have a `tool_run_id`.
- Avoid duplicating this logic inside the logging decorator to prevent double-inserts of artifacts.

### 3.4 — avoid re-querying source data
The format tool must NOT re-run the underlying data query.
Instead:
- It should read the already logged raw output from Supabase:
  - `tool_runs.id` (UUID) is the canonical identifier for the raw data payload.
  - This is called `source_tool_run_id`.
- The format tool reads:
  - `tool_runs.response_json` using `source_tool_run_id`
- Then it formats deterministically and stores artifacts.

---

## 4) Supabase Schema Requirements

### 4.1 Existing: `tool_runs`
- Has primary key `id uuid`
- Logs `session_id`, `turn_id`, `tool_name`, `request_json`, `response_json`, etc.
- Tool responses already include `meta.tool_run_id` (the tool run id).

**Note about logging semantics (current system)**
- The `tool_runs` table uses a CHECK constraint that does not allow a transient `running` status.
- Therefore, tool logging writes a **terminal row** with `status = success|error` (no “start row + update”).

### 4.2 New: `artifacts` table
We need a new Supabase table `public.artifacts` with:

- `id uuid primary key default gen_random_uuid()`
- `session_id text not null`
- `turn_id int not null`
- `artifact_type text not null`  
  Example values: `presentation_table`, `raw_table`, `chart`, `markdown`
- `title text null`
- `created_mode text not null default 'auto_default'`  
  Example values: `auto_default`, `interpret_request`, `manual`
- `source_tool_run_id uuid null references tool_runs(id)`
- `source_tool_name text null` (convenience, redundant)
- `parent_artifact_id uuid null references artifacts(id)` (optional, for reformat lineage)
- `format_spec jsonb not null default '{}'`
- `payload jsonb not null default '{}'`
- `row_count int null`
- `bytes int null`
- timestamps: `created_at`, `updated_at` + updated_at trigger
- indexes:
  - `(session_id, turn_id, created_at desc)`
  - `(session_id, created_at desc)`
  - `(source_tool_run_id)`
  - `(artifact_type)`

(We already created a SQL DDL for this, but Codex should ensure it exists and matches usage.)

---

## 5) Tool + API Behavior Requirements

### 5.1 Data tools (example: income_statement)
We want to extend the request model for income statement with an optional field:

- `format_request: Optional[str] = None`

This field is a free-text request from user/agent like:
- “i mkr, top 5, sort desc”
- “visa i tusental”
- “sortera i storleksordning”
- “filtrera till intäkter”

### 5.2 Auto-trigger logic after income_statement
When `/tools/income-statement` finishes and returns response:
- it already logged a tool_run row and has `tool_run_id` (either returned in meta or otherwise known)
- backend should call formatting pipeline:

If `format_request` is empty/None:
- `mode = auto_default`  
- Use `default_spec` for income_statement (no LLM)

If `format_request` is present:
- `mode = interpret_request`  
- Use LLM to generate a `format_spec` then apply deterministically

The format tool should then create a `presentation_table` artifact and insert it into `public.artifacts`.

**Critical requirement (no re-query)**
- Even though `format_request` lives on the data tool request (e.g. income_statement), the formatting step must always load the raw data via:
  - `source_tool_run_id = tool_run_id`
  - fetch `tool_runs.response_json` from Supabase
- Do not reuse in-memory table objects; treat `tool_runs.response_json` as the canonical source of truth.

### 5.3 Format tool callable independently
We also want the format tool to be callable (later or now) as an endpoint, e.g.:

`POST /tools/format`

Inputs:
- `session_id`
- `turn_id`
- `source_tool_run_id`
- `format_request` (optional)
- `mode` (optional; if omitted: choose based on format_request)
- `source_tool_name` (optional; used to pick default_spec)

Outputs:
- artifact id
- maybe a summary of applied spec

Even if not exposed immediately, code should be structured so it can be exposed later.

---

## 6) `format_spec` (v1) — Minimal Structured Spec

The LLM (when used) must produce a structured spec with strict validation. Keep v1 minimal.

Suggested `format_spec` fields (v1):

- `unit`: `"sek" | "tsek" | "msek"` (canonical)
  - Accept `"mkr"` as an alias and normalize to `"msek"`
- `decimals`: integer (0–3)
- `top_n`: integer (e.g., 5, 10, 30) optional
- `sort`: list of `{ "col": string, "dir": "asc"|"desc" }` optional
- `filters`: optional dict or list of conditions, but v1 can limit to simple equality filters:
  - e.g. `{ "konto_typ": ["intäkter"] }`
- `include_totals`: boolean optional
- `derive`: optional list of derived columns operations (v1 compute)

**v1 simplification (income statement)**
- Income statement output is a wide table with period columns (e.g. `"2025-02"`, `"2025-03"`).
- For v1, if `sort`/`top_n` is requested and no explicit value column is given:
  - use the **latest period** in `meta.periods` as the default sort key.
- Keep `filters` limited to **dimension columns only** (not period columns).

### 6.1 v1 compute/derive limitation
We allow only derived columns that can be computed from the already returned table (no extra data sources).

Examples allowed:
- `diff = colA - colB`
- `pct = diff / colB` with safe div0 handling
- `abs(col)`
- `share_of_total` (if total exists or can be computed from the same table)

The resolver must validate:
- referenced columns exist
- operation is allowed
- output type and name are safe
- caps on number of derived columns (e.g., <= 5)

---

## 7) Artifact Payload Standard (UI-agnostic)

For `artifact_type = presentation_table`, standardize payload to a UI-neutral schema:

```json
{
  "kind": "table",
  "columns": ["rr_level_1", "amount_mkr"],
  "rows": [
    {"rr_level_1": "Såld vård internt", "amount_mkr": 77.1}
  ],
  "format": {
    "unit": "mkr",
    "decimals": 1,
    "sorted_by": "amount_mkr desc",
    "row_limit": 5
  }
}
```

Notes:
- Keep row/column counts bounded (v1: max 100 rows, max 12 columns)
- If the source data has > 100 rows, truncate and add a small note in the payload, e.g.:
  - `"notes": ["Source had 437 rows; showing first 100 rows."]`
- Store the exact applied `format_spec` separately in `artifacts.format_spec`
- Keep `payload` ready for Streamlit and future Lovable rendering.

**Clarification: raw vs formatted schemas**
- Raw tool outputs (stored in `tool_runs.response_json`) can have tool-specific structures (e.g. `{"columns": [...], "table": [...]}`).
- Formatted artifacts (stored in `artifacts.payload`) should always follow the UI-neutral schema above (`columns` + `rows` + `format` + optional `notes`).

---

## 8) Lineage Requirements

Each artifact row should include:

- `session_id`, `turn_id`
- `source_tool_run_id` (points to raw data tool run)
- `format_spec` (the normalized spec used)
- optional `parent_artifact_id` if this is a reformat of a previous artifact
- optional `created_mode`

This ensures traceability and reproducibility.

---

## 9) Implementation Tasks (High-Level)

1) Ensure Supabase `public.artifacts` table exists (DDL above).
2) Add `format_request` optional field to income statement request schema.
3) Implement formatting module in backend:
   - load raw response JSON from `tool_runs` by `source_tool_run_id`
   - generate `format_spec`:
     - default_spec (no LLM) OR
     - LLM interpretation (only when `format_request` is non-empty)
   - apply deterministic resolver (pandas-based) to produce `presentation_table` payload
   - insert artifacts into Supabase
4) Add backend auto-trigger after `/tools/income-statement` completes (v1 only tool):
   - choose mode based on presence of `format_request`
   - call formatting module using `tool_run_id`
5) (Optional later) expose `/tools/format` endpoint for explicit reformatting requests.
6) Update UI to show both:
   - raw table artifact (existing) AND
   - formatted/presentation_table artifact when available
   - allow browsing by `turn_id`.

---

## 10) Constraints / Non-Goals (for now)

- No “full arbitrary compute” (no extra queries, no joins, no new data sources)
- No expensive additional agent round-trips for every tool call
- Avoid sending large raw tables back into the LLM (keep LLM work minimal)
- Avoid coupling solution tightly to Streamlit; ensure portability to Lovable

---

## 11) Notes on RelevanceAI Integration

- RelevanceAI agent calls my FastAPI endpoints as tools.
- The auto-trigger formatting should happen in backend; RelevanceAI just triggers the data tool call.
- Format tool may later be exposed as a separate tool for agent to call when the user asks “format this differently”.

---

## 12) What Codex Should Do Next

Given my codebase, implement the above with minimal changes:
- add artifacts schema (if not already applied)
- add `format_request` to income statement request model and endpoint
- implement formatting module with default_spec and deterministic resolver
- insert artifacts into Supabase
- integrate auto-trigger after income statement tool run
- keep changes generic and modular (works for future tools too)

End.
