"""
Persistent session storage – saves/loads chat history to sessions.json.
Plotly figures are serialised with fig.to_json() / pio.from_json().
Pandas DataFrames are serialised with df.to_json() / pd.read_json().
"""

import io
import json
import os
import uuid
from datetime import datetime

import pandas as pd
import plotly.io as pio

SESSIONS_FILE = "sessions.json"


# ── serialise / deserialise a single history entry ────────────────────────────

def _serialise_result(result):
    if result is None:
        return {"type": "none"}
    if isinstance(result, pd.DataFrame):
        return {"type": "dataframe", "data": result.to_json(orient="split")}
    return {"type": "scalar", "data": str(result)}


def _deserialise_result(blob):
    t = blob.get("type", "none")
    if t == "none":
        return None
    if t == "dataframe":
        return pd.read_json(io.StringIO(blob["data"]), orient="split")
    return blob.get("data")


def _serialise_entry(entry: dict) -> dict:
    fig = entry.get("fig")
    return {
        "query":       entry.get("query", ""),
        "answer":      entry.get("answer", entry.get("insight", "")),
        "description": entry.get("description", ""),
        "code":        entry.get("code", ""),
        "result":      _serialise_result(entry.get("result")),
        "fig":         fig.to_json() if fig is not None else None,
    }


def _deserialise_entry(raw: dict) -> dict:
    fig_json = raw.get("fig")
    fig = pio.from_json(fig_json) if fig_json else None
    return {
        "query":       raw.get("query", ""),
        "answer":      raw.get("answer", raw.get("insight", "")),
        "description": raw.get("description", ""),
        "code":        raw.get("code", ""),
        "result":      _deserialise_result(raw.get("result", {"type": "none"})),
        "fig":         fig,
    }


# ── file I/O ──────────────────────────────────────────────────────────────────

def _load_all() -> dict:
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all(data: dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── public API ────────────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    """Return sessions sorted newest-first: [{id, title, timestamp}, ...]"""
    data = _load_all()
    sessions = [
        {"id": sid, "title": s["title"], "timestamp": s["timestamp"]}
        for sid, s in data.items()
    ]
    return sorted(sessions, key=lambda s: s["timestamp"], reverse=True)


def save_session(session_id: str, title: str, history: list):
    """Persist (or overwrite) a session."""
    data = _load_all()
    data[session_id] = {
        "title":     title,
        "timestamp": datetime.now().isoformat(),
        "history":   [_serialise_entry(e) for e in history],
    }
    _save_all(data)


def load_session(session_id: str) -> list:
    """Return the deserialised history list for a session."""
    data = _load_all()
    raw = data.get(session_id, {})
    return [_deserialise_entry(e) for e in raw.get("history", [])]


def delete_session(session_id: str):
    data = _load_all()
    data.pop(session_id, None)
    _save_all(data)


def new_session_id() -> str:
    return str(uuid.uuid4())


def make_title(first_query: str, max_len: int = 45) -> str:
    title = first_query.strip()
    return title if len(title) <= max_len else title[:max_len - 1] + "…"
