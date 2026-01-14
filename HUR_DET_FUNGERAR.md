## Finance AI Agent — Hur det fungerar

Det här dokumentet beskriver hur din Finance AI Agent fungerar end-to-end, med extra fokus på **formateringssystemet** (presentation artifacts).

### 1) Byggblock (komponenter)

- **Streamlit UI**: `ui/app.py`
  - Chat + “Artifacts”-vy per `turn_id`
  - Kan också göra “reformat” direkt (utan agenten) via `POST /tools/format`
- **FastAPI backend**: `app/main.py`
  - Agent-proxy-endpoints: `/agent/chat`, `/agent/poll`
  - Tool-endpoints: `/tools/*` (t.ex. P&L, variance, format)
  - UI-endpoints: `/ui/tool-runs`, `/ui/artifacts`, `/ui/latest-turn`
- **RelevanceAI agent**
  - Orkestrerar verktygskörningar och genererar assistentsvar
  - Anropar dina FastAPI tools
- **Supabase/Postgres**
  - Lagrar loggade tool-körningar i `tool_runs`
  - Lagrar UI-vänliga “presentation artifacts” i `artifacts`

### 2) Nyckel-ID:n: `session_id` och `turn_id`

Hela systemet bygger på deterministisk korrelation:

- **`session_id: str`**: stabilt id för en UI-session / konversationstråd
- **`turn_id: int`**: räknare som ökar per användarmeddelande

De här används för att:

- koppla chat-turn → tool calls → sparade loggar
- hämta rätt artifacts för en viss turn

### 3) Chatflöde (agent)

#### 3.1 Starta/fortsätt en agent-task

- UI skickar `POST /agent/chat` med `{session_id, turn_id, message, conversation_id?}`
- Backend prependerar en meta-rad till agenten:
  - `"[meta] session_id=... turn_id=...\n\n{message}"`
- Backend försöker sedan hitta rätt turn-svar deterministiskt via Relevance “steps”
  - om det inte blir klart inom ~25s returneras `status="running"` och UI pollar vidare

#### 3.2 Pollning

- UI anropar `GET /agent/poll?conversation_id=...&since_ts=...&session_id=...&turn_id=...`
- Backend letar efter ett agentmeddelande som matchar meta-raden för aktuell `turn_id`

### 4) Tool-körningar och loggning (`tool_runs`)

Varje tool-endpoint i FastAPI är dekorerad med `@log_tool_call(...)` i `app/tool_logging.py`.

Det gör att varje tool-körning:

- **loggas** som en terminal rad i Supabase `tool_runs` med `status="success"` eller `status="error"`
- får `request_json`, `response_json` (eller `error_json`)
- får `duration_ms`, `row_count` (best-effort), `bytes`
- (om svaret är ett dict) får även `meta.tool_run_id` injicerat i tool-svaret

### 5) Artifacts: rådata vs presentation

Systemet skiljer på två typer av “saker UI kan visa”:

- **Råa tool outputs**: kommer från `tool_runs.response_json`
- **Presentation artifacts**: sparas i `artifacts` (ex. `artifact_type="presentation_table"`)

UI hämtar:

- råa tool runs via `GET /ui/tool-runs`
- presentation artifacts via `GET /ui/artifacts` (vanligtvis filtrerat på `artifact_type=presentation_table`)

## Formatering (detaljerad)

Formateringssystemet är byggt för att vara:

- **deterministiskt** (ingen re-query av data; formatering är en ren transform på redan loggad output)
- **UI-neutralt** (payload är generell JSON som kan visas i Streamlit eller annan frontend)
- **inkrementellt** (nya format-önskemål patchas på tidigare spec per turn)

### 6) Två sätt att skapa en presentation_table

#### 6.1 Auto-format (backend)

I `app/tool_logging.py` auto-skapas en presentation artifact för vissa verktyg (v1: `income_statement_tool`):

- Efter att `tool_runs` har skrivits och vi har ett `tool_run_id`
- Kallas formateringspipen best-effort (får aldrig krascha tool-svaret)
- Resultatet sparas i `artifacts` som `presentation_table`

#### 6.2 Manuell reformat (via `/tools/format`)

Endpoint: `POST /tools/format` i `app/main.py`.

Request-modell: `app/schemas/format_tool.py`:

- `session_id`, `turn_id`
- `source_tool_run_id` (UUID i `tool_runs`; “källan” man vill formatera)
- antingen:
  - `format_spec` (struktur) **eller**
  - `format_request` (fri text, tolkas via LLM)
- `reset` (starta om från default)

### 7) FormatSpec (vad som kan styras)

`FormatSpec` är en Pydantic-modell i `app/formatting/format_spec.py`. Den är medvetet minimal och har skyddsräcken (caps) för att hålla v1 stabil.

#### 7.1 Centrala fält

- **`unit`**: `"sek" | "tsek" | "msek"`
  - aliases normaliseras (t.ex. `"kr"→"sek"`, `"mkr"→"msek"`)
- **`decimals`**: 0–3 (avrundning i presentationsteget)
- **`top_n`**: 1–100 (begränsa antal rader)
- **`sort`**: lista av `{col, dir}` där `dir` är `asc|desc`
  - `col` kan vara `null` → resolver väljer default (se 7.2)
- **`include_totals`**: om totalsrader ska behållas
- **`column_decimals`**: per-kolumn-avrundning (0–3) + *skalbara selectors* (se 9.6)
- **`rename_columns`**: mapping `{old_col: new_display_name}` (presentation-only)
- **`derive`**: lista av härledda kolumner (beräknas från tabellen, utan re-query)
- **`filters`**: AND-filter (legacy)
- **`filter_groups`**: grupper med `op=and|or` (top-level AND mellan grupper)
- **`filter_expr`**: nästlad boolean-trädmodell (v2) som *tar precedence* över filters/filter_groups

#### 7.2 Default spec

`default_format_spec()` bygger en default som matchar typiska finanstabeller:

- `unit="sek"`
- `decimals=0`
- `include_totals=True`
- `sort=[{col: null, dir: "desc"}]` vilket innebär: **sortera fallande på “högerkolumnen”** (oftast senaste period)

Resolvern `resolve_missing_sort_columns(...)` sätter `sort.col` till `columns[-1]` om den saknas.

### 8) Inkrementell spec-resolver (viktig)

Funktionen `resolve_incremental_format_spec(...)` i `app/formatting/spec_pipeline.py` bygger den “effektiva” spec:en när man reformaterar.

#### 8.1 Bas-spec (per turn)

- Basen är **senaste** `presentation_table` för samma `(session_id, turn_id)` om den finns
- annars används `default_format_spec()`

#### 8.2 “Apply on latest presentation data”

När `/tools/format` körs med `apply_on_latest_presentation_data=True`:

- om det redan finns en `presentation_table` i den turnen, används dess `source_tool_run_id` som *effektiv källa*
- det gör “reformat på senaste presentationen” naturligt i UI

#### 8.3 Reset

Reset kan komma från två håll:

- request-fältet `reset=true`
- eller att LLM tolkar texten som reset (t.ex. “nollställ”, “reset”, “default”)

Om reset triggas: basen blir `default_format_spec()` innan partial changes appliceras.

#### 8.4 Merge-semantik för nested fält

Det här är en stor poäng i din implementation: vissa fält “mår dåligt” av vanlig overwrite, så pipen har special-merge:

- **`rename_columns`**: dict-merge (nya keys skriver över gamla)
- **`column_decimals`**: dict-merge
- **`derive`**: merge-by-name (ny derive med samma `name` skriver över gamla)
- **`filters`**: merge-by-`id` om id finns, annars “signature” (`col|op|value`)
- **`filter_groups`**: merge-by-`id` om id finns, annars signature
- **`filter_expr`**: replace-semantik (om satt → ersätter)

### 9) LLM-tolkning av `format_request` (OpenAI-kompatibelt)

LLM-tolkningen finns i `app/formatting/interpret_format_request_openai.py` och används när `format_request` är en fri text.

#### 9.1 Kontext byggs från tool_run (utan rader)

För att hålla kostnad och risk låg skickas **inte** tabellrader till modellen.

`_build_context_from_tool_run(...)` hämtar från `tool_runs`:

- `tool_name`
- `available_columns` (kolumnlista, ev. härledd från första raden)
- `dimension_columns` (från `meta.rows` om det finns)
- `default_sort_column` (högerkolumnen)
- `totals_marker`

#### 9.2 OpenAI Chat Completions

Anrop sker mot en OpenAI-kompatibel endpoint:

- `OPENAI_API_KEY` krävs
- `OPENAI_BASE_URL` kan override:as (default OpenAI)
- `OPENAI_MODEL` kan override:as (default i koden: `"gpt-4o-mini"`)
- `temperature=0`
- systemprompt kräver **JSON-only output** och begränsar till en tillåten nyckellista

#### 9.3 Deterministiska “hints” (robusthet)

Efter att LLM returnerat JSON gör koden deterministiska extractioner från texten och patchar in om modellen missade dem:

- **rename**: t.ex. “döp kolumn rr_level_1 till Resultaträkning”
- **derive**: t.ex. “skillnad mellan 2025-02 och 2025-01”
- **reset**: “nollställ/återställ/reset/default…”
- **column_decimals**: “värdekolumner 1 decimal”, “kolumn 2025-02: 2 decimal”
- **filters**: enkla svensk/eng-textmönster (eq/neq/contains samt numeriska jämförelser)
- **OR-logik**: om användaren skriver “eller/or” och modellen gav “flat filters” kan det tolkas om till `filter_groups` för att undvika att OR oavsiktligt blir AND

#### 9.4 Best-effort validering och notes

LLM-responsen behandlas som **partial spec**:

- `apply_partial_format_overrides(...)` försöker applicera bara det som validerar
- ogiltiga delar ignoreras, och läggs som **notes** (användarfeedback)

Det här gör att systemet hellre “gör inget” än att krascha.

#### 9.5 Guardrails för “unsupported”

Vissa önskemål är medvetet “not supported” i v1 (ex. kolumnreordering/lägg till/ta bort på generellt sätt). Sådant fångas upp och blir notes.

#### 9.6 column_decimals: selectors

Din formatterare stödjer “skalbara selectors” i `column_decimals` som expanderas vid rendering:

- **`__VALUE__`**: alla value-kolumner (t.ex. periodkolumner)
- **`__PCT__`**: derived percent-kolumner (t.ex. derive med `pct_change`, identifierade via derive-name)
- **`re:<pattern>`**: regex på kolumnnamn (t.ex. `re:^2025-`)

### 10) Deterministisk formatterare: från tool_run → payload

Kärnan är `build_presentation_table_payload_from_tool_run(...)` i `app/formatting/presentation_table.py`.

#### 10.1 Inläsning och tabellmodell

- Hämtar `tool_runs.response_json` via `source_tool_run_id`
- Kräver att `response_json` är ett objekt med:
  - `table: list[dict]` (rader)
  - `columns: list[str]` (valfritt; annars härleds från första raden)
  - `meta: dict` (valfritt; används för dims/totals)

#### 10.2 Ordning på transformsteg (viktigt)

Formatteringen kör i följande ordning:

1. **Infer columns** + dela upp i **dims** och **value_cols**
2. **Totals-hantering**: om `include_totals=false` droppas totalsrader
3. **Unit conversion** på value-kolumner (SEK→tsek/msek)
4. **Sort-kolumn resolver** (om `sort.col` saknas → högerkolumn)
5. **Derive** (beräknade kolumner)
   - sker *efter unit conversion* men *före rounding* för stabil procent-matematik
6. **Filter**
   - om `filter_expr` finns: använd den
   - annars: använd `filters` / `filter_groups`
   - om både `filters` och `filter_groups` finns: `filters` konverteras till en explicit AND-grupp och läggs först
7. **Sort** (ignorerar okända sortkolumner och noterar det)
8. **top_n** (om satt)
9. **Bounds**: max 100 rader, max 12 kolumner (v1 säkerhetscap)
10. **Rounding** (presentation-only)
    - kan styras per-kolumn via expanderade `column_decimals`
11. **rename_columns** (presentation-only)
    - körs sist så att spec alltid refererar till ursprungliga kolumnnamn
12. Bygg en UI-neutral payload + en deterministisk svensk sammanfattning (`summary_sv`, `steps_sv`)
13. Tagga totalsrader (`row_tags`) och byt totals-marker mot en mänsklig label

#### 10.3 Output: UI-neutral payload

Payloaden ser i grova drag ut så här:

```json
{
  "kind": "table",
  "columns": ["rr_level_1", "2025-01", "2025-02", "diff"],
  "rows": [ { "...": "..." } ],
  "format": {
    "unit": "mkr",
    "unit_canonical": "msek",
    "decimals": 1,
    "sorted_by": "2025-02 desc",
    "row_limit": 5,
    "include_totals": true,
    "summary_sv": "Enhet: mkr. Decimaler: 1. ...",
    "steps_sv": ["..."],
    "row_tags": [["total"], [], ...]
  },
  "notes": ["Applied top_n=5.", "..."]
}
```

### 11) Lagring: singleton-upsert + lineage

När payloaden är byggd skapas en artifact-row och skrivs till `artifacts`.

Funktionen `upsert_singleton_presentation_artifact(...)` i `app/formatting/presentation_table.py` ser till att:

- det finns **exakt en** `presentation_table` per `(session_id, turn_id)`
- om man reformaterar:
  - samma `source_tool_run_id` + samma `format_spec` → ingen write (`mode="unchanged"`)
  - bara notes ändras → uppdaterar notes utan lineage (`mode="notes_update"`)
  - annars → overwrite i-place och sparar föregående version i `payload.lineage` (bounded)

Det ger en stabil UI-upplevelse: “en artifact per turn”, men ändå historik.

### 12) UI: hur det används i praktiken

I `ui/app.py`:

- Råa körningar visas från `/ui/tool-runs`
- Presentation artifacts visas från `/ui/artifacts?artifact_type=presentation_table`
- UI renderar payloaden om `payload.kind == "table"`

UI har även en “Reformat”-panel som kan:

- skapa en ny presentation via `POST /tools/format`
- antingen med strukturerad spec (unit/decimals/top_n/…) eller med fri text (`format_request`)
- nollställa formatering via `reset=true`

### 13) Exempel: format_request → effekt

- **“i mkr, 1 decimal, top 5”**
  - `unit="msek"`, `decimals=1`, `top_n=5`
  - sort blir fortfarande “högerkolumn desc” om sort-kolumn inte nämns

- **“sortera asc på 2025-01”**
  - `sort=[{"col":"2025-01","dir":"asc"}]`
  - om kolumnen inte finns → ignoreras med en note

- **“visa bara rr_level_1 = Intäkter eller rr_level_1 = Kostnader”**
  - preferens mot `filter_groups` med `op="or"` så att det inte blir AND av misstag

---

## Tips om du vill bygga vidare

- För fler UI-format: håll dig till att lägga metadata i `payload["format"]` (UI-neutralt)
- För mer avancerade filter: bygg vidare på `filter_expr` (den har redan size-guardrails)
- Om du vill stödja fler “fria text”-intents: lägg till deterministiska regex-hints först, och låt LLM fylla luckor


