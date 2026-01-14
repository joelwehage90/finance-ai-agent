# Finance AI Agent

A finance-focused AI agent that combines a Streamlit chat UI, a FastAPI backend, a RelevanceAI agent (LLM orchestration + tool calls), and Supabase/Postgres (financial data + observability/logging). The system supports multi-turn chat, deterministic correlation of tool outputs to chat turns, and an “artifacts” view that lets you inspect tool-run inputs/outputs per turn.

---

## 1) High-level architecture

**Components**
- **Streamlit UI** (`ui/app.py`)
  - Chat interface
  - Generates and maintains `session_id` (string) and `turn_id` (int)
  - Sends messages to backend and polls for agent responses
  - Fetches and renders “artifacts” (tool run logs) per `turn_id`
- **FastAPI backend** (`app/main.py`)
  - Exposes agent endpoints (`/agent/chat`, `/agent/poll`)
  - Exposes tool endpoints (`/tools/*`)
  - Exposes UI helper endpoint (`/ui/tool-runs`) to query Supabase tool logs
  - Protected with an API key header
- **RelevanceAI agent**
  - Orchestrates reasoning and tool selection
  - Calls FastAPI tool endpoints
  - Produces final assistant responses
- **Supabase/Postgres**
  - Stores financial data tables/views used by tools
  - Stores tool-run logs (`tool_runs`) for observability and UI artifacts

**Deployment**
- Local dev: `uvicorn` (FastAPI) + `streamlit run` (UI)
- Production: FastAPI deployed on Render (UI can remain local or be deployed separately)

---

## 2) End-to-end flow (deterministic)

### A) Session/Turn identity
- `session_id: str` — stable per UI session (unique per conversation thread)
- `turn_id: int` — increments for each user message in the UI

These IDs are the primary correlation keys:
- UI ↔ backend (agent endpoints)
- RelevanceAI ↔ backend (tool endpoints)
- backend ↔ Supabase (`tool_runs`)
- UI ↔ Supabase (via backend `/ui/tool-runs`)

### B) Chat request → agent response
1. **User writes a message** in Streamlit UI.
2. UI increments `turn_id` and sends payload to backend:
   - `POST /agent/chat` with `{session_id, turn_id, message, conversation_id?}`
3. Backend passes message to RelevanceAI (includes meta line with `session_id` and `turn_id`).
4. RelevanceAI may call one or more tools (`/tools/*` endpoints).
5. Backend polls RelevanceAI task state and returns:
   - `status: completed|running|error`
   - `assistant_message` when available
   - `conversation_id` for polling continuity
6. UI shows assistant response; if still running, UI polls:
   - `GET /agent/poll?conversation_id=...`

### C) Tool calls → logging → artifacts
When RelevanceAI calls a tool endpoint:
1. FastAPI tool endpoint receives a request that includes `session_id` and `turn_id`.
2. A decorator wraps each tool function to:
   - Execute the tool logic (Supabase queries, aggregation, pandas transforms, etc.)
   - Insert a **terminal** record into Supabase `tool_runs` with `status=success|error`
     (the current `tool_runs` schema uses a CHECK constraint and does not allow a transient `running` state)
   - Add `tool_run_id` into the returned payload under `meta.tool_run_id` (when the tool returns a dict)
3. UI fetches artifacts for the current turn:
   - `GET /ui/tool-runs?session_id=...&turn_id=...`
4. UI renders artifacts (request/response JSON, metadata, optional tables).

---

## 3) Data model (Supabase)

### A) Financial tables/views (analysis data)
Common entities (example names):
- `transactions` — raw transaction-level data
- `transactions_view` — curated/derived view with dimensions (often used for grouping and filtering)
- `pnl_monthly_rr` — pre-aggregated P&L view by month and RR hierarchy
- Supporting tables:
  - `account_mapping`
  - `business_definitions`
  - `metric_definitions`

### B) Observability / tool logging
- `tool_runs` (or equivalent) logs tool invocations:
  - `session_id` (text)
  - `turn_id` (int)
  - `tool_name` (text)
  - `status` (success/error)
  - `request_json` (jsonb)
  - `response_json` (jsonb)
  - `error_json` (jsonb)
  - `duration_ms`, `row_count`, `bytes`, timestamps

This provides:
- Debugging/observability
- Ability to re-render artifacts per turn
- Deterministic traceability from chat question → tool output → final answer

### C) Artifacts (optional, v1)
You can also store “rendered artifacts” in a dedicated `artifacts` table (e.g. curated tables/charts/markdown):
- `session_id`, `turn_id` (correlation)
- `artifact_type`, `title`, `created_mode`
- `source_tool_run_id` (FK to `tool_runs.id`)
- `format_spec` + `payload` (jsonb)

This is useful if you want the UI to show a curated view rather than raw `tool_runs.response_json`.

---

## 4) Tooling (FastAPI tool endpoints)

### 1) Income Statement Tool (`/tools/income-statement`)
**Purpose**
- Returns a P&L table for one or more periods
- Supports isolated month or YTD accumulation

**Typical inputs**
- `session_id`, `turn_id`
- `compare_mode`: `"month"` | `"ytd"`
- `periods`: list like `["YYYY-MM", ...]`
- `rows`: grouping dimensions (e.g., `["konto_typ","rr_level_1"]`)
- `filters`: optional dimension filters
- `include_total`: bool

**Output**
- `meta`: source, periods, filters, format hints
- `columns`: list of columns
- `table`: list of records (rows)

**Data source**
- Primarily `pnl_monthly_rr` for performance

---

### 2) Variance Tool (`/tools/variance`)
**Purpose**
- Variance between base and comparison period
- Supports “grain” dimensions and top positive/negative drivers

**Typical inputs**
- `session_id`, `turn_id`
- `compare_mode`
- `base_period`, `comp_period`
- `grain`: list of dimensions
- `filters`
- `top_n_pos`, `top_n_neg`

**Output**
- Often multiple tables (e.g., top_pos, top_neg, totals)
- Returned as JSON records

---

### 3) Definitions Tool (`/tools/definitions`)
**Purpose**
- Lookup of business and metric definitions

**Typical inputs**
- `terms` list and/or `search` string
- include flags (`include_business`, `include_metrics`)
- `limit`

**Data source**
- `business_definitions`, `metric_definitions`

---

### 4) Account Mapping Tool (`/tools/account-mapping`)
**Purpose**
- Map accounts to RR hierarchy levels and categories
- Support hierarchy mode or lookup mode

**Typical inputs**
- `mode`: `"hierarchy"` | `"lookup"`
- `accounts` or `rr_level_*` filters
- `search`, `limit`

**Data source**
- `account_mapping`

---

## 5) Key conventions

### A) session_id + turn_id are first-class
- Every tool request includes `session_id` and `turn_id`
- Every tool call is logged using these keys
- UI uses these keys to retrieve artifacts per chat turn

### B) Separation of concerns
- Streamlit: presentation, local session state, polling, artifacts rendering
- FastAPI: auth, agent proxy, tool endpoints, Supabase access, logging
- RelevanceAI: reasoning, tool selection, response generation
- Supabase: data + observability records

### C) Spec-first agent behavior (global)
This repo is designed around a consistent “spec → map → execute” pattern across *all* tools:

- **Spec layer (allowed shape)**: each tool has a strict schema (Pydantic / JSON-ish) defining inputs, constraints, defaults, and allowed enums.
- **LLM mapping layer (interpretation)**: the LLM receives the tool spec + relevant runtime context (available columns/dimensions/periods, etc.) and maps free-text into a **partial spec** (only fields the user explicitly asked to change), plus **notes** for ambiguity.
- **Execution layer (deterministic)**: the backend merges the partial spec with defaults, validates/normalizes, and executes deterministically.

Policy rules:
- **No hardcoded semantic mappings**: avoid hardcoding “user phrase X always means column/field Y”. Let the LLM decide from spec + context; if unclear, add notes (or ask a follow-up question if it’s blocking).
- **LLM outputs partial specs only**: omit keys the user did not request to change.
- **Backend owns defaults + validation**: defaults and validation happen in code, not via the LLM.
- **Best-effort + notes**: apply what can be inferred; record what couldn’t be interpreted instead of failing silently.

### D) Tool output contract (for formatting + UI artifacts)
If you want a tool’s output to be compatible with the deterministic formatter (`/tools/format` → `presentation_table` artifacts), the tool run’s `response_json` must follow one of these shapes:

**Single-table (recommended for most tools)**
- **Required**
  - `meta: { ... }`
  - `table: list[dict]` (rows)
- **Strongly recommended**
  - `columns: list[str]` (stable column order for UI; important because JSONB can reorder keys)
- **Optional formatting hints (UI-neutral, used by formatter/UI)**
  - `meta.rows: list[str]` — dimension columns (left side)
  - `meta.value_columns: list[str]` — value columns (right side)
  - `meta.column_formats: { [col]: { kind: "percent", scale: "fraction", decimals?: int } }`
  - `meta.default_sort: list[{ col: str, dir: "asc"|"desc" }]` — applied when user hasn’t provided an explicit sort
  - Totals convention: totals rows can be marked with `__TOTAL__` in dim columns; formatter will tag them and display “Totalt”.

**Multi-table (one tool run → one artifact with multiple tables)**
- `meta: { ... }` (shared hints for all tables)
- Each table is a key in the top-level dict with value `list[dict]`, e.g.:
  - `kostnader_pos: [...]`
  - `kostnader_neg: [...]`
  - `intakter_pos: [...]`
  - `intakter_neg: [...]`
- Avoid mixing other non-table keys except `meta` and optional debug keys prefixed with `_` (these are ignored by the formatter).

Notes:
- The formatter will always try to honor `columns` / `meta.rows + meta.value_columns` for stable ordering.
- Percent/fraction columns should be declared in `meta.column_formats` so UIs (including Lovable) can display `23%` while storing values as `0.23`.

---

## 6) Local development setup

### A) Environment / secrets
You typically need:
- FastAPI API key (shared by UI)
- Supabase URL + service key (or anon key if RLS is configured accordingly)
- RelevanceAI API key + agent id

**Streamlit secrets**
Create:
- `ui/.streamlit/secrets.toml`

Example:
```toml
API_BASE="http://127.0.0.1:8000"
API_KEY="some-long-random-string"
```

**FastAPI env vars**
Create a `.env` in the repo root (or set environment variables):
- `API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `DB_URI` (Postgres read-only connection string, used by `/tools/nl_sql`)
- `NL_SQL_STATEMENT_TIMEOUT_MS` (optional, default 10000)
- `RAI_API_KEY`
- `RAI_REGION`
- `RAI_PROJECT`
- `RAI_AGENT_ID`
- `OPENAI_API_KEY` (used by LLM-backed tools, incl. `/tools/nl_sql`)
- `OPENAI_MODEL` (optional)
- `OPENAI_BASE_URL` (optional, for OpenAI-compatible providers)

### B) Run locally
From the repo root:

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In another terminal:

```bash
.\.venv\Scripts\streamlit run ui/app.py
```

Open Streamlit at `http://localhost:8501`.

---

## 7) Notes / troubleshooting

### A) RelevanceAI steps parsing when tools run
When tools run, Relevance may emit step items like `content.type="tool-run"`. Some SDK versions try to parse steps into strict models and can fail.
This backend uses a raw JSON step fetch to avoid that and keep `/agent/poll` reliable.

### B) Artifacts panel behavior
Streamlit “sticky CSS” is not reliable across versions (due to internal scroll containers). The UI supports pinning Artifacts into the sidebar, which is the most stable option.

---

## 8) Deployment (Render)
- Ensure Render uses **Python 3.12.x** (this repo pins `3.12.7` via `.python-version`)
- Set the env vars listed above in Render
- Start command example: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
