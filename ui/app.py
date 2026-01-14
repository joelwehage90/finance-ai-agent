import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd

# ----------------------------
# Config (prefer Streamlit secrets, fallback to env)
# ----------------------------
API_BASE = st.secrets.get("API_BASE", os.getenv("API_BASE", "http://127.0.0.1:8000"))
API_KEY  = st.secrets.get("API_KEY",  os.getenv("API_KEY", ""))

# Header name: båda funkar i praktiken, men håll dig till X-API-Key för tydlighet
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


st.set_page_config(page_title="Finance AI Agent", layout="wide")
st.sidebar.caption(f"API_BASE: {API_BASE}")
st.sidebar.caption(f"API_KEY loaded: {'YES' if API_KEY else 'NO'}")
try:
    # /health is intentionally unauthenticated in the backend
    _h = requests.get(f"{API_BASE}/health", timeout=5).json()
    if isinstance(_h, dict):
        st.sidebar.caption(f"Backend /health: {_h.get('status', 'ok')}")
    else:
        st.sidebar.caption("Backend /health: ok")
except Exception as e:
    st.sidebar.error(f"Backend /health unreachable: {e}")


# ----------------------------
# Helpers
# ----------------------------
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _raise_for_bad_response(r: requests.Response, *, method: str, url: str):
    """
    Streamlit UX: visa alltid statuskod + serverns feltext (t.ex. FastAPI HTTPException.detail).
    requests.raise_for_status() tappar ofta body i Streamlit.
    """
    if r.status_code < 400:
        return

    body_text = ""
    try:
        body_text = r.text or ""
    except Exception:
        body_text = ""

    detail = None
    try:
        j = r.json()
        if isinstance(j, dict):
            detail = j.get("detail")
    except Exception:
        pass

    msg = f"{method} {url} failed ({r.status_code} {r.reason})"
    if detail:
        msg += f"\n\ndetail: {detail}"
    if body_text and (not detail or str(detail) not in body_text):
        clipped = body_text if len(body_text) <= 2000 else (body_text[:2000] + "…")
        msg += f"\n\nresponse: {clipped}"
    raise RuntimeError(msg)


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=120)
    except requests.RequestException as e:
        raise RuntimeError(f"POST {url} failed (network error): {e}") from e
    _raise_for_bad_response(r, method="POST", url=url)
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}


def api_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=120)
    except requests.RequestException as e:
        raise RuntimeError(f"GET {url} failed (network error): {e}") from e
    _raise_for_bad_response(r, method="GET", url=url)
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}


def ensure_state():
    st.session_state.setdefault("session_id", f"session_{int(time.time())}")
    st.session_state.setdefault("turn_id", 0)
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("messages", [])  # list of dicts: {role,text,turn_id,ts}
    st.session_state.setdefault("pending", None)  # {conversation_id, since_ts, session_id, turn_id, user_text, created_ts}
    st.session_state.setdefault("artifacts_by_turn", {})  # turn_id -> list[tool_run]
    st.session_state.setdefault("formatted_by_turn", {})  # turn_id -> list[artifact]
    st.session_state.setdefault("selected_turn", None)
    st.session_state.setdefault("last_fetch_error", None)  # str|None
    # Streamlit input clearing: rotate widget key to reliably reset text_area
    st.session_state.setdefault("input_key", 0)


def append_message(role: str, text: str, turn_id: Optional[int] = None):
    st.session_state["messages"].append(
        {"role": role, "text": text, "turn_id": turn_id, "ts": _now_str()}
    )

def reset_chat_input():
    """Clear the chat input reliably by rotating the widget key."""
    k = f"input_text_{st.session_state.get('input_key', 0)}"
    if k in st.session_state:
        del st.session_state[k]
    st.session_state["input_key"] = int(st.session_state.get("input_key", 0)) + 1


def fetch_tool_runs(session_id: str, turn_id: Optional[int] = None) -> List[Dict[str, Any]]:
    params = {"session_id": session_id}
    if turn_id is not None:
        params["turn_id"] = int(turn_id)
    try:
        data = api_get("/ui/tool-runs", params=params)
        st.session_state["last_fetch_error"] = None
    except Exception as e:
        st.session_state["last_fetch_error"] = f"/ui/tool-runs failed: {e}"
        raise
    # Backend returns {"runs": [...]} (older versions used "tool_runs")
    return (data.get("runs") or data.get("tool_runs") or [])  # type: ignore[return-value]


def fetch_formatted_artifacts(
    session_id: str,
    turn_id: int,
    *,
    artifact_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "session_id": session_id,
        "turn_id": int(turn_id),
        "include_payload": True,
        "include_format_spec": True,
    }
    if artifact_type:
        params["artifact_type"] = artifact_type
    try:
        data = api_get("/ui/artifacts", params=params)
        st.session_state["last_fetch_error"] = None
    except Exception as e:
        st.session_state["last_fetch_error"] = f"/ui/artifacts failed: {e}"
        raise
    return (data.get("artifacts") or [])  # type: ignore[return-value]


def _render_presentation_payload(payload: Any):
    """
    Render the v1 UI-neutral presentation payload:
      {"kind":"table","columns":[...],"rows":[...],"format":{...},"notes":[...]}
    """
    if not payload:
        st.caption("Tomt payload.")
        return

    if isinstance(payload, dict) and payload.get("kind") == "table":
        rows = payload.get("rows") or []
        cols = payload.get("columns") or []
        fmt = payload.get("format") or {}
        notes = payload.get("notes") or []

        # Human-readable formatting summary (preferred UI view)
        if isinstance(fmt, dict):
            summary_sv = fmt.get("summary_sv")
            steps_sv = fmt.get("steps_sv")
            if isinstance(summary_sv, str) and summary_sv.strip():
                st.markdown(f"**Formatering (sammanfattning):** {summary_sv.strip()}")
            if isinstance(steps_sv, list) and steps_sv:
                steps_clean = [str(s).strip() for s in steps_sv if isinstance(s, str) and str(s).strip()]
                if steps_clean:
                    with st.expander("Formateringssteg", expanded=True):
                        st.markdown("\n".join([f"- {s}" for s in steps_clean]))

        if isinstance(fmt, dict) and fmt:
            with st.expander("Format", expanded=False):
                st.json(fmt)

        if isinstance(notes, list) and notes:
            with st.expander("Notes", expanded=bool(len(notes) <= 3)):
                for n in notes:
                    if isinstance(n, str) and n.strip():
                        st.caption(n.strip())

        # Display-only formatting: if backend marks columns as percent fractions, render as "23.4%".
        rows_to_show = rows
        try:
            if isinstance(fmt, dict) and isinstance(fmt.get("column_formats"), dict) and isinstance(rows, list) and rows:
                cfmt = fmt.get("column_formats") or {}
                percent_cols = []
                for k, v in cfmt.items():
                    if not isinstance(v, dict):
                        continue
                    if v.get("kind") == "percent" and v.get("scale") == "fraction":
                        d = v.get("decimals")
                        try:
                            d_i = int(d)
                        except Exception:
                            d_i = 0
                        percent_cols.append((str(k), max(0, min(3, d_i))))
                if percent_cols:
                    out_rows = []
                    for r in rows:
                        if not isinstance(r, dict):
                            out_rows.append(r)
                            continue
                        r2 = dict(r)
                        for col, d_i in percent_cols:
                            val = r2.get(col)
                            try:
                                if val is None:
                                    continue
                                f = float(val)
                                r2[col] = format(f, f".{d_i}%")
                            except Exception:
                                # leave as-is
                                pass
                        out_rows.append(r2)
                    rows_to_show = out_rows
        except Exception:
            pass

        # Honor backend-provided column order if present (important because JSONB may reorder keys).
        try:
            df = pd.DataFrame(rows_to_show if isinstance(rows_to_show, list) else [])
            if isinstance(cols, list) and cols:
                cols_clean = [str(c) for c in cols if c is not None]
                desired = [c for c in cols_clean if c in df.columns]
                extras = [c for c in df.columns if c not in set(desired)]
                df = df[desired + extras]
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            # Fallback: original behavior
            st.dataframe(rows_to_show, use_container_width=True, hide_index=True)
        return

    if isinstance(payload, dict) and payload.get("kind") == "multi_table":
        fmt = payload.get("format") or {}
        notes = payload.get("notes") or []
        tables = payload.get("tables") or []

        # Top-level summary/steps
        if isinstance(fmt, dict):
            summary_sv = fmt.get("summary_sv")
            steps_sv = fmt.get("steps_sv")
            if isinstance(summary_sv, str) and summary_sv.strip():
                st.markdown(f"**Formatering (sammanfattning):** {summary_sv.strip()}")
            if isinstance(steps_sv, list) and steps_sv:
                steps_clean = [str(s).strip() for s in steps_sv if isinstance(s, str) and str(s).strip()]
                if steps_clean:
                    with st.expander("Formateringssteg", expanded=True):
                        st.markdown("\n".join([f"- {s}" for s in steps_clean]))

        if isinstance(notes, list) and notes:
            with st.expander("Notes", expanded=bool(len(notes) <= 3)):
                for n in notes:
                    if isinstance(n, str) and n.strip():
                        st.caption(n.strip())

        if not isinstance(tables, list) or not tables:
            st.caption("Inga tabeller i multi_table payload.")
            return

        labels = []
        table_dicts = []
        for t in tables:
            if not isinstance(t, dict):
                continue
            k = t.get("table_key") or t.get("key") or ""
            k = str(k) if k is not None else ""
            label = k.replace("_", " ").strip() or "Tabell"
            labels.append(label)
            table_dicts.append(t)

        if not table_dicts:
            st.caption("Inga giltiga tabeller i multi_table payload.")
            return

        tab_objs = st.tabs(labels)
        for tab, t in zip(tab_objs, table_dicts):
            with tab:
                _render_presentation_payload(t)
        return

    st.json(payload)


def render_formatted_artifacts(artifacts: List[Dict[str, Any]]):
    if not artifacts:
        st.caption("Inga presentation artifacts för valt turn-id.")
        return
    for a in artifacts:
        with st.expander(
            f"{a.get('artifact_type','artifact')} • {a.get('created_mode','')} • turn_id={a.get('turn_id','')}",
            expanded=False,
        ):
            st.write("**created_at:**", a.get("created_at"))
            st.write("**title:**", a.get("title") or "-")
            st.write("**source_tool_name:**", a.get("source_tool_name") or "-")
            st.write("**source_tool_run_id:**", a.get("source_tool_run_id") or "-")
            st.write("**payload:**")
            _render_presentation_payload(a.get("payload"))
            # Optional: show lineage history if present (singleton overwrite mode)
            payload = a.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("lineage"), list) and payload.get("lineage"):
                with st.expander("Lineage (tidigare versioner)", expanded=False):
                    for i, it in enumerate(reversed(payload.get("lineage") or []), start=1):
                        if not isinstance(it, dict):
                            continue
                        st.write(f"**#{i}** • created_at={it.get('created_at') or '-'} • mode={it.get('created_mode') or '-'}")
                        st.write("source_tool_run_id:", it.get("source_tool_run_id") or "-")
                        if it.get("format_spec") is not None:
                            st.json(it.get("format_spec"))
            if isinstance(payload, dict) and isinstance(payload.get("redo_lineage"), list) and payload.get("redo_lineage"):
                with st.expander("Redo lineage (framåt)", expanded=False):
                    for i, it in enumerate(reversed(payload.get("redo_lineage") or []), start=1):
                        if not isinstance(it, dict):
                            continue
                        st.write(f"**#{i}** • created_at={it.get('created_at') or '-'} • mode={it.get('created_mode') or '-'}")
                        st.write("source_tool_run_id:", it.get("source_tool_run_id") or "-")
                        if it.get("format_spec") is not None:
                            st.json(it.get("format_spec"))


def _pick_default_reformat_source(
    formatted_artifacts: List[Dict[str, Any]],
    tool_runs: List[Dict[str, Any]],
    *,
    source_tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Determine best defaults for reformat:
    - Prefer latest presentation artifact for this tool (gives parent_artifact_id + source_tool_run_id)
    - Otherwise, fall back to latest successful tool run for this tool
    """
    # Prefer latest presentation artifact in this turn (+ tool, when specified)
    if formatted_artifacts:
        candidates = []
        for a in formatted_artifacts:
            if not isinstance(a, dict):
                continue
            if source_tool_name and a.get("source_tool_name") != source_tool_name:
                continue
            candidates.append(a)
        if candidates:
            last = candidates[-1]
            return {
                "source_tool_run_id": last.get("source_tool_run_id"),
                "parent_artifact_id": last.get("id"),
                "format_spec": last.get("format_spec"),
            }

    # Fallback: latest successful run for this tool (if known)
    if source_tool_name:
        for tr in reversed(tool_runs or []):
            if not isinstance(tr, dict):
                continue
            if tr.get("tool_name") == source_tool_name and tr.get("status") == "success":
                return {"source_tool_run_id": tr.get("id"), "parent_artifact_id": None}

    # Legacy fallback: latest successful income statement tool run
    for tr in reversed(tool_runs or []):
        if not isinstance(tr, dict):
            continue
        if tr.get("tool_name") == "income_statement_tool" and tr.get("status") == "success":
            return {"source_tool_run_id": tr.get("id"), "parent_artifact_id": None}

    # Final fallback: any successful tool run
    for tr in reversed(tool_runs or []):
        if not isinstance(tr, dict):
            continue
        if tr.get("status") == "success" and tr.get("id"):
            return {"source_tool_run_id": tr.get("id"), "parent_artifact_id": None}

    return {"source_tool_run_id": None, "parent_artifact_id": None}


def render_reformat_panel(
    *,
    session_id: str,
    turn_id: int,
    source_tool_name: str,
    formatted_artifacts: List[Dict[str, Any]],
    tool_runs: List[Dict[str, Any]],
):
    """
    UI-only convenience: create a new presentation_table via POST /tools/format.
    Works without the agent.
    """
    source_tool_name = str(source_tool_name or "").strip()
    defaults = _pick_default_reformat_source(formatted_artifacts, tool_runs, source_tool_name=source_tool_name or None)
    current_spec = defaults.get("format_spec") if isinstance(defaults, dict) else None
    if not isinstance(current_spec, dict):
        current_spec = {}
    _unit_opts = ["sek", "tsek", "msek"]
    _unit_val = str(current_spec.get("unit") or "sek")
    _unit_idx = _unit_opts.index(_unit_val) if _unit_val in _unit_opts else 0
    _decimals_val = int(current_spec.get("decimals") or 0)
    _top_n_val = int(current_spec.get("top_n") or 0) if current_spec.get("top_n") is not None else 0
    _include_totals_val = bool(current_spec.get("include_totals")) if current_spec.get("include_totals") is not None else True

    with st.expander("Reformat (skapa ny presentation artifact)", expanded=False):
        st.caption("Detta anropar `/tools/format` och skapar en ny artifact utan att re-query:a data.")

        # NOTE: Widgets inside st.form don't rerun on change until submit.
        # If we conditionally render text inputs based on a checkbox inside the form,
        # the text box never appears when you click the checkbox. Therefore:
        # - keep the checkbox + text area OUTSIDE the form
        # - keep the submit button INSIDE the form
        use_free_text = st.checkbox(
            "Använd fritext (LLM)",
            value=False,
            key=f"use_free_text_turn_{turn_id}_{source_tool_name}",
        )
        format_request_val = st.text_area(
            "format_request (fri text)",
            value="",
            height=70,
            disabled=not use_free_text,
            help='Ex: "msek en decimal", "i mkr, top 5", "sort desc".',
            key=f"format_request_turn_{turn_id}_{source_tool_name}",
        )
        if use_free_text:
            st.caption("LLM tolkar format_request. Backend bygger själv kontext från tool_run (kolumner, rightmost, totals).")

        # Undo/Redo (persistent via artifact payload lineage/redo_lineage)
        # NOTE: This MUST be rendered AFTER the free-text widgets. Otherwise, clicking Undo/Redo triggers
        # st.rerun() before the widgets are instantiated in that run, which can cause Streamlit to drop
        # widget state and leave the text_area disabled until re-toggled.
        active_art = formatted_artifacts[-1] if formatted_artifacts else {}
        active_payload = active_art.get("payload") if isinstance(active_art, dict) else {}
        if not isinstance(active_payload, dict):
            active_payload = {}
        undo_depth = len(active_payload.get("lineage") or []) if isinstance(active_payload.get("lineage"), list) else 0
        redo_depth = len(active_payload.get("redo_lineage") or []) if isinstance(active_payload.get("redo_lineage"), list) else 0

        c_undo, c_redo, c_meta = st.columns([1, 1, 3])
        with c_undo:
            if st.button(
                "Undo",
                key=f"undo_fmt_turn_{turn_id}_{source_tool_name}",
                use_container_width=True,
                disabled=undo_depth == 0,
                help="Gå tillbaka till föregående formatering (utan att re-query:a data).",
            ):
                try:
                    with st.spinner("Undo..."):
                        api_post(
                            "/tools/format/undo",
                            {"session_id": session_id, "turn_id": int(turn_id), "source_tool_name": source_tool_name},
                        )
                    st.session_state["formatted_by_turn"][turn_id] = fetch_formatted_artifacts(
                        session_id, turn_id=turn_id, artifact_type="presentation_table"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Undo misslyckades: {e}")
        with c_redo:
            if st.button(
                "Redo",
                key=f"redo_fmt_turn_{turn_id}_{source_tool_name}",
                use_container_width=True,
                disabled=redo_depth == 0,
                help="Gå framåt till nästa formatering efter en undo.",
            ):
                try:
                    with st.spinner("Redo..."):
                        api_post(
                            "/tools/format/redo",
                            {"session_id": session_id, "turn_id": int(turn_id), "source_tool_name": source_tool_name},
                        )
                    st.session_state["formatted_by_turn"][turn_id] = fetch_formatted_artifacts(
                        session_id, turn_id=turn_id, artifact_type="presentation_table"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Redo misslyckades: {e}")
        with c_meta:
            if undo_depth or redo_depth:
                st.caption(f"Undo-steg: {undo_depth} • Redo-steg: {redo_depth}")

        # Reset button (applies to latest presentation_table for this turn)
        if st.button(
            "Nollställ formatering",
            key=f"reset_format_turn_{turn_id}_{source_tool_name}",
            use_container_width=True,
        ):
            source_tool_run_id_reset = str(defaults.get("source_tool_run_id") or "").strip()
            if not source_tool_run_id_reset:
                st.error("source_tool_run_id saknas (kan inte reset:a).")
            else:
                try:
                    with st.spinner("Nollställer formatering..."):
                        resp = api_post(
                            "/tools/format",
                            {
                                "session_id": session_id,
                                "turn_id": int(turn_id),
                                "source_tool_run_id": source_tool_run_id_reset,
                                "reset": True,
                                "created_mode": "manual",
                                "title": "Presentation (reset)",
                            },
                        )
                    st.success(f"Nollställde formatering: {resp.get('artifact_id')}")
                    try:
                        st.session_state["formatted_by_turn"][turn_id] = fetch_formatted_artifacts(
                            session_id, turn_id=turn_id, artifact_type="presentation_table"
                        )
                    except Exception:
                        pass
                    st.rerun()
                except Exception as e:
                    st.error(f"Reset misslyckades: {e}")

        with st.form(key=f"reformat_form_turn_{turn_id}_{source_tool_name}"):
            source_tool_run_id = st.text_input(
                "source_tool_run_id",
                value=str(defaults.get("source_tool_run_id") or ""),
                help="UUID från tool_runs.id (källan som ska formateras).",
                key=f"source_tool_run_id_turn_{turn_id}_{source_tool_name}",
            )

            unit_label = st.selectbox(
                "Unit",
                options=["sek", "tsek", "msek"],
                index=_unit_idx,
                help="sek=kr, tsek=tkr, msek=mkr",
                disabled=use_free_text,
                key=f"unit_turn_{turn_id}_{source_tool_name}",
            )
            decimals = st.slider(
                "Decimals",
                min_value=0,
                max_value=3,
                value=_decimals_val,
                disabled=use_free_text,
                key=f"decimals_turn_{turn_id}_{source_tool_name}",
            )
            top_n = st.number_input(
                "Top N (0 = ingen)",
                min_value=0,
                max_value=100,
                value=_top_n_val,
                step=1,
                disabled=use_free_text,
                key=f"top_n_turn_{turn_id}_{source_tool_name}",
            )
            include_totals = st.checkbox(
                "Include totals",
                value=_include_totals_val,
                disabled=use_free_text,
                key=f"include_totals_turn_{turn_id}_{source_tool_name}",
            )
            title = st.text_input(
                "Title (valfri)",
                value="Presentation (reformat)",
                key=f"title_turn_{turn_id}_{source_tool_name}",
            )

            submitted = st.form_submit_button("Skapa reformat", use_container_width=True)

        if submitted:
            if not source_tool_run_id.strip():
                st.error("source_tool_run_id krävs.")
                return

            payload: Dict[str, Any] = {
                "session_id": session_id,
                "turn_id": int(turn_id),
                "source_tool_run_id": source_tool_run_id.strip(),
                "created_mode": "manual",
                "title": title.strip() or "Presentation (reformat)",
            }
            if use_free_text and format_request_val and format_request_val.strip():
                payload["format_request"] = format_request_val.strip()
            # When using LLM, send ONLY format_request; defaults + best-effort LLM fields will apply in backend.
            if not use_free_text:
                spec: Dict[str, Any] = {"unit": unit_label, "decimals": int(decimals), "include_totals": bool(include_totals)}
                if int(top_n) > 0:
                    spec["top_n"] = int(top_n)
                payload["format_spec"] = spec

            try:
                with st.spinner("Skapar reformat..."):
                    resp = api_post("/tools/format", payload)
                if resp.get("mode") == "unchanged":
                    st.info(f"Ingen ändring (återanvände artifact): {resp.get('artifact_id')}")
                else:
                    st.success(f"Uppdaterade artifact: {resp.get('artifact_id')}")
                # Refresh cached artifacts for this turn
                try:
                    st.session_state["formatted_by_turn"][turn_id] = fetch_formatted_artifacts(
                        session_id, turn_id=turn_id, artifact_type="presentation_table"
                    )
                except Exception:
                    pass
                st.rerun()
            except Exception as e:
                st.error(f"Reformat misslyckades: {e}")


def _tool_label(tool_name: str) -> str:
    """
    Human-friendly label for tabs.
    """
    t = str(tool_name or "").strip()
    if not t:
        return "Okänt verktyg"
    # keep names readable and stable
    return t.replace("_tool", "").replace("_", " ").strip().title()


def render_presentation_tabs(
    *,
    session_id: str,
    turn_id: int,
    formatted_artifacts: List[Dict[str, Any]],
    tool_runs: List[Dict[str, Any]],
):
    """
    Render presentation artifacts grouped by source_tool_name as tabs.
    """
    # Build tabs from:
    # - tools that already have a presentation_table artifact
    # - tools that have a successful raw tool_run in this turn (so user can create the first presentation tab)
    #
    # Preserve order of appearance: artifacts first (already ordered), then tool_runs order.
    tool_names: List[str] = []
    seen: set[str] = set()

    for a in formatted_artifacts or []:
        if not isinstance(a, dict):
            continue
        tn = str(a.get("source_tool_name") or "").strip() or "unknown"
        if tn not in seen:
            seen.add(tn)
            tool_names.append(tn)

    for tr in tool_runs or []:
        if not isinstance(tr, dict):
            continue
        if tr.get("status") != "success":
            continue
        tn = str(tr.get("tool_name") or "").strip()
        if not tn:
            continue
        if tn not in seen:
            seen.add(tn)
            tool_names.append(tn)

    if not tool_names:
        st.caption("Inga presentation artifacts eller lyckade tool runs för valt turn-id.")
        return

    tab_labels = [_tool_label(t) for t in tool_names]
    tabs = st.tabs(tab_labels)
    for tab, tool_name in zip(tabs, tool_names):
        with tab:
            arts = [
                a
                for a in (formatted_artifacts or [])
                if isinstance(a, dict) and (str(a.get("source_tool_name") or "").strip() or "unknown") == tool_name
            ]
            render_reformat_panel(
                session_id=session_id,
                turn_id=int(turn_id),
                source_tool_name=tool_name,
                formatted_artifacts=arts,
                tool_runs=tool_runs,
            )
            render_formatted_artifacts(arts)

def _render_tool_response(response_json: Any):
    """
    Render common tool outputs:
    - income_statement_tool: {"columns": [...], "table": [records...]}
    - variance_tool: {"kostnader_pos": [...], ...}
    - definitions_tool: {"items": [...]}
    - account_mapping_tool: {"items": [...]}
    Fallback: show JSON.
    """
    if not response_json:
        st.caption("Tomt svar.")
        return

    if isinstance(response_json, dict):
        # Income statement: columns + table
        if isinstance(response_json.get("table"), list):
            table = response_json.get("table") or []
            cols = response_json.get("columns")
            if isinstance(cols, list) and cols:
                st.dataframe(table, use_container_width=True, hide_index=True)
            else:
                st.dataframe(table, use_container_width=True, hide_index=True)
            return

        # Variance: multiple named tables
        variance_keys = ["kostnader_pos", "kostnader_neg", "intakter_pos", "intakter_neg"]
        if any(isinstance(response_json.get(k), list) for k in variance_keys):
            tabs = [k for k in variance_keys if isinstance(response_json.get(k), list)]
            if tabs:
                tab_objs = st.tabs(tabs)
                for i, k in enumerate(tabs):
                    with tab_objs[i]:
                        st.dataframe(response_json.get(k) or [], use_container_width=True, hide_index=True)
                return

        # Generic items list
        if isinstance(response_json.get("items"), list):
            st.dataframe(response_json.get("items") or [], use_container_width=True, hide_index=True)
            return

    # Fallback
    st.json(response_json)


def render_artifacts(tool_runs: List[Dict[str, Any]]):
    if not tool_runs:
        st.info("Inga artifacts/loggade tool-runs för valt turn-id.")
        return

    for tr in tool_runs:
        with st.expander(
            f"{tr.get('tool_name','tool')} • {tr.get('status','')} • turn_id={tr.get('turn_id','')}",
            expanded=False,
        ):
            st.write("**created_at:**", tr.get("created_at"))
            st.write("**session_id:**", tr.get("session_id"))
            st.write("**turn_id:**", tr.get("turn_id"))
            st.write("**request_json:**")
            if tr.get("request_json") is not None:
                st.json(tr.get("request_json"))
            else:
                st.caption("(none)")

            st.write("**response_json:**")
            _render_tool_response(tr.get("response_json"))

            err = tr.get("error_json")
            if err:
                st.write("**error_json:**")
                st.json(err)


# ----------------------------
# UI
# ----------------------------
ensure_state()

st.sidebar.header("Session")
st.sidebar.text_input("session_id", key="session_id")
st.sidebar.write("conversation_id:", st.session_state.get("conversation_id") or "-")

if st.sidebar.button("Ny session", use_container_width=True):
    st.session_state["session_id"] = f"session_{int(time.time())}"
    st.session_state["turn_id"] = 0
    st.session_state["conversation_id"] = None
    st.session_state["messages"] = []
    st.session_state["pending"] = None
    st.session_state["artifacts_by_turn"] = {}
    st.session_state["selected_turn"] = None
    st.session_state["input_key"] = int(st.session_state.get("input_key", 0)) + 1
    st.rerun()

st.sidebar.divider()

# Turn navigator (för artifacts)
turn_ids = sorted({m.get("turn_id") for m in st.session_state["messages"] if m.get("turn_id") is not None})
if turn_ids:
    default_turn = st.session_state["selected_turn"]
    if default_turn not in turn_ids:
        default_turn = turn_ids[-1]
    st.sidebar.selectbox(
        "Välj turn-id (artifacts)",
        options=turn_ids,
        index=turn_ids.index(default_turn),
        key="selected_turn",
    )
else:
    st.sidebar.caption("Inga turn ännu (chatten används inte / agent saknas).")
    st.sidebar.number_input(
        "Ange turn-id manuellt (artifacts)",
        min_value=1,
        step=1,
        key="selected_turn_manual",
    )
    if st.sidebar.button("Ladda artifacts", use_container_width=True):
        st.session_state["selected_turn"] = int(st.session_state.get("selected_turn_manual") or 1)
        st.rerun()

auto_poll = st.sidebar.checkbox("Auto-poll pending", value=True)
pin_artifacts_sidebar = st.sidebar.checkbox("Fäst Artifacts i sidopanelen", value=True)
hide_format_tool_runs = st.sidebar.checkbox("Raw tool runs: dölj format_tool", value=True)

# Always allow manual navigation even if chat turn_ids exist (useful after restarts)
st.sidebar.divider()
st.sidebar.caption("Snabbnavigering (artifacts)")
st.sidebar.number_input(
    "Gå till turn-id",
    min_value=1,
    step=1,
    value=int(st.session_state.get("selected_turn") or (turn_ids[-1] if turn_ids else 1) or 1),
    key="goto_turn_id",
)
cols_nav = st.sidebar.columns([1, 1])
if cols_nav[0].button("Gå", use_container_width=True):
    st.session_state["selected_turn"] = int(st.session_state.get("goto_turn_id") or 1)
    st.rerun()
if cols_nav[1].button("Senaste (backend)", use_container_width=True):
    try:
        data = api_get("/ui/latest-turn", {"session_id": st.session_state["session_id"]})
        latest = data.get("latest_turn_id")
        if latest is not None:
            st.session_state["selected_turn"] = int(latest)
            st.rerun()
        else:
            st.sidebar.warning("Backend hittade ingen turn för session_id ännu.")
    except Exception as e:
        st.sidebar.error(f"Kunde inte hämta senaste turn: {e}")

# Layout: chat + artifacts
col_chat, col_art = st.columns([2, 1])

with col_chat:
    st.title("Finance AI Agent")
    st.caption(f"API_BASE={API_BASE}")

    # Render messages
    for m in st.session_state["messages"]:
        role = m["role"]
        if role == "user":
            st.markdown(f"**Du (turn {m.get('turn_id')}):** {m['text']}")
        else:
            st.markdown(f"**Agent:** {m['text']}")

    st.divider()

    user_text = st.text_area(
        "Skriv ett meddelande",
        height=100,
        key=f"input_text_{st.session_state.get('input_key', 0)}",
    )

    cols = st.columns([1, 1, 1, 1])
    send_clicked = cols[0].button("Skicka", use_container_width=True)
    poll_clicked = cols[1].button("Poll pending now", use_container_width=True)
    debug_clicked = cols[2].button("Debug metadata", use_container_width=True)

    if send_clicked:
        if not user_text.strip():
            st.warning("Skriv ett meddelande först.")
        else:
            st.session_state["turn_id"] += 1
            turn_id = st.session_state["turn_id"]

            append_message("user", user_text.strip(), turn_id=turn_id)

            payload = {
                "session_id": st.session_state["session_id"],
                "turn_id": turn_id,
                "conversation_id": st.session_state.get("conversation_id"),
                "message": user_text.strip(),
            }

            try:
                with st.spinner("Skickar till agent..."):
                    resp = api_post("/agent/chat", payload)
            except Exception as e:
                st.error(f"Agent-chat misslyckades: {e}")
                # Keep app usable for artifacts browsing even if agent is down/unauthorized
                st.session_state["pending"] = None
                reset_chat_input()
                st.rerun()

            st.session_state["conversation_id"] = resp.get("conversation_id") or st.session_state.get("conversation_id")

            if resp.get("status") == "completed":
                assistant_msg = resp.get("assistant_message", "")
                # Store message with timestamp for tracking
                msg_data = {"role": "assistant", "text": assistant_msg, "turn_id": turn_id, "ts": _now_str()}
                if resp.get("message_ts"):
                    msg_data["message_ts"] = resp.get("message_ts")
                st.session_state["messages"].append(msg_data)
                try:
                    tool_runs = fetch_tool_runs(st.session_state["session_id"], turn_id=turn_id)
                    st.session_state["artifacts_by_turn"][turn_id] = tool_runs
                except Exception:
                    pass
                try:
                    formatted = fetch_formatted_artifacts(
                        st.session_state["session_id"], turn_id=turn_id, artifact_type="presentation_table"
                    )
                    st.session_state["formatted_by_turn"][turn_id] = formatted
                except Exception:
                    pass
                # Clear input reliably
                reset_chat_input()
                if "selected_turn" in st.session_state:
                    del st.session_state["selected_turn"]
                # Set the new value after deletion
                st.session_state["selected_turn"] = turn_id
                st.rerun()
            else:
                st.session_state["pending"] = {
                    "conversation_id": st.session_state["conversation_id"],
                    "since_ts": resp.get("since_ts"),  # REQUIRED
                    "session_id": st.session_state["session_id"],
                    "turn_id": turn_id,
                    "user_text": user_text.strip(),
                    "created_ts": time.time(),
                }
                # Clear input reliably
                reset_chat_input()
                st.rerun()

    # Debug metadata
    if debug_clicked:
        conv_id = st.session_state.get("conversation_id")
        if not conv_id:
            st.warning("No conversation_id available. Start a conversation first.")
        else:
            try:
                with st.spinner("Fetching debug info..."):
                    debug_data = api_get("/agent/debug-metadata", {"conversation_id": conv_id})
                
                st.subheader("Debug Information")
                st.json(debug_data)
                
                # Show extracted candidate
                if debug_data.get("extracted_candidate"):
                    st.success("✅ Found response in metadata!")
                    st.write("**Extracted text:**")
                    st.code(debug_data["extracted_candidate"].get("text", ""))
                else:
                    st.warning("⚠️ No response extracted from metadata")
                
                # Show state
                st.write(f"**State:** {debug_data.get('state', 'unknown')}")
                
                # Show steps info
                if debug_data.get("steps_info", {}).get("has_steps"):
                    st.info(f"Found {debug_data['steps_info']['results_count']} items in steps")
                    if debug_data["steps_info"].get("sample_result"):
                        st.write("**Sample step result:**")
                        st.json(debug_data["steps_info"]["sample_result"])
                
            except Exception as e:
                st.error(f"Debug failed: {e}")

    # Poll logic
    pending = st.session_state.get("pending")
    if pending:
        st.info(f"Pending turn_id={pending['turn_id']} • since_ts={pending.get('since_ts')}")
        
        # Always poll if button clicked, or if auto_poll is enabled and not too old (5 minutes max)
        max_age_seconds = 300  # 5 minutes
        age_seconds = time.time() - pending.get("created_ts", 0)
        do_poll = poll_clicked or (auto_poll and age_seconds < max_age_seconds)

        if do_poll:
            try:
                params = {
                    "conversation_id": pending["conversation_id"],
                    "since_ts": pending.get("since_ts") or 0,
                    "session_id": pending.get("session_id") or st.session_state.get("session_id"),
                    "turn_id": pending.get("turn_id"),
                    "include_meta": False,
                    "include_preview": False,
                    "include_steps": False,
                    "include_debug": False,
                }
                with st.spinner("Pollar efter svar..."):
                    resp = api_get("/agent/poll", params=params)

                if resp.get("status") == "completed":
                    turn_id = pending["turn_id"]
                    assistant_msg = resp.get("assistant_message", "")
                    # Store the message with its timestamp for tracking
                    msg_data = {"role": "assistant", "text": assistant_msg, "turn_id": turn_id, "ts": _now_str()}
                    if resp.get("message_ts"):
                        msg_data["message_ts"] = resp.get("message_ts")
                    st.session_state["messages"].append(msg_data)
                    st.session_state["pending"] = None
                    try:
                        tool_runs = fetch_tool_runs(st.session_state["session_id"], turn_id=turn_id)
                        st.session_state["artifacts_by_turn"][turn_id] = tool_runs
                    except Exception:
                        pass
                    try:
                        formatted = fetch_formatted_artifacts(
                            st.session_state["session_id"], turn_id=turn_id, artifact_type="presentation_table"
                        )
                        st.session_state["formatted_by_turn"][turn_id] = formatted
                    except Exception:
                        pass
                    # Delete key before setting to avoid widget modification error
                    if "selected_turn" in st.session_state:
                        del st.session_state["selected_turn"]
                    st.session_state["selected_turn"] = turn_id
                    st.rerun()
                elif resp.get("status") == "error":
                    append_message("assistant", "Task failed.", turn_id=pending["turn_id"])
                    st.session_state["pending"] = None
                    st.rerun()
                else:
                    # Status is "running" - continue polling if auto_poll is enabled
                    if auto_poll:
                        time.sleep(1.0)
                        st.rerun()
            except Exception as e:
                st.error(f"Error polling agent: {e}")
                # Still rerun to retry, but with a delay
                if auto_poll:
                    time.sleep(2.0)
                    st.rerun()
        else:
            # Too old or auto_poll disabled - show message
            if age_seconds >= max_age_seconds:
                st.warning(f"Pending request is too old ({int(age_seconds)}s). Click 'Poll pending now' to check status.")
            elif not auto_poll:
                st.info("Auto-poll is disabled. Click 'Poll pending now' to check status.")

with col_art:
    st.subheader("Artifacts")
    sel_turn = st.session_state.get("selected_turn")
    if sel_turn is None:
        if st.session_state["artifacts_by_turn"]:
            sel_turn = sorted(st.session_state["artifacts_by_turn"].keys())[-1]

    if sel_turn is not None:
        if sel_turn not in st.session_state["artifacts_by_turn"]:
            try:
                st.session_state["artifacts_by_turn"][sel_turn] = fetch_tool_runs(
                    st.session_state["session_id"], turn_id=sel_turn
                )
            except Exception:
                st.session_state["artifacts_by_turn"][sel_turn] = []
        if sel_turn not in st.session_state["formatted_by_turn"]:
            try:
                st.session_state["formatted_by_turn"][sel_turn] = fetch_formatted_artifacts(
                    st.session_state["session_id"], turn_id=sel_turn, artifact_type="presentation_table"
                )
            except Exception:
                st.session_state["formatted_by_turn"][sel_turn] = []
        if st.session_state.get("last_fetch_error"):
            st.error(st.session_state["last_fetch_error"])
            st.caption("Om du ser 401 Unauthorized: se till att Streamlit `API_KEY` matchar backend `API_KEY`.")
        if pin_artifacts_sidebar:
            st.caption("Artifacts är fästa i sidopanelen till vänster.")
        else:
            st.markdown("**Presentation artifacts**")
            render_presentation_tabs(
                session_id=st.session_state["session_id"],
                turn_id=int(sel_turn),
                formatted_artifacts=st.session_state["formatted_by_turn"].get(sel_turn, []),
                tool_runs=st.session_state["artifacts_by_turn"].get(sel_turn, []),
            )
            st.divider()
            st.markdown("**Raw tool runs**")
            raw_runs = st.session_state["artifacts_by_turn"].get(sel_turn, [])
            if hide_format_tool_runs:
                raw_runs = [tr for tr in raw_runs if isinstance(tr, dict) and tr.get("tool_name") != "format_tool"]
            render_artifacts(raw_runs)
    else:
        st.caption("Inga artifacts ännu.")

# Render pinned artifacts in the sidebar (this is the most reliable “sticky” area in Streamlit).
if pin_artifacts_sidebar:
    with st.sidebar:
        st.divider()
        st.subheader("Artifacts")
        sel_turn = st.session_state.get("selected_turn")
        if sel_turn is None:
            if st.session_state["artifacts_by_turn"]:
                sel_turn = sorted(st.session_state["artifacts_by_turn"].keys())[-1]
        if sel_turn is not None:
            if sel_turn not in st.session_state["artifacts_by_turn"]:
                try:
                    st.session_state["artifacts_by_turn"][sel_turn] = fetch_tool_runs(
                        st.session_state["session_id"], turn_id=sel_turn
                    )
                except Exception:
                    st.session_state["artifacts_by_turn"][sel_turn] = []
            if sel_turn not in st.session_state["formatted_by_turn"]:
                try:
                    st.session_state["formatted_by_turn"][sel_turn] = fetch_formatted_artifacts(
                        st.session_state["session_id"], turn_id=sel_turn, artifact_type="presentation_table"
                    )
                except Exception:
                    st.session_state["formatted_by_turn"][sel_turn] = []

            st.markdown("**Presentation artifacts**")
            render_presentation_tabs(
                session_id=st.session_state["session_id"],
                turn_id=int(sel_turn),
                formatted_artifacts=st.session_state["formatted_by_turn"].get(sel_turn, []),
                tool_runs=st.session_state["artifacts_by_turn"].get(sel_turn, []),
            )
            st.divider()
            st.markdown("**Raw tool runs**")
            raw_runs = st.session_state["artifacts_by_turn"].get(sel_turn, [])
            if hide_format_tool_runs:
                raw_runs = [tr for tr in raw_runs if isinstance(tr, dict) and tr.get("tool_name") != "format_tool"]
            render_artifacts(raw_runs)
        else:
            st.caption("Inga artifacts ännu.")
