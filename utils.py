import pandas as pd
import json
import sqlite3
import tempfile
import os
import time


def extract_text(response) -> str:
    """Gemini models sometimes return response.content as a list of parts
    (e.g. thinking + text blocks) instead of a plain string. This normalizes
    either shape into a single string."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts).strip()
    return str(content).strip()


def invoke_with_retry(llm, messages, max_retries: int = 3, base_delay: float = 2.0):
    """Retries an LLM call on transient errors (503 overloaded, 429 rate limit)
    with exponential backoff. Raises the error if all retries are exhausted."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            is_transient = "503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str
            if not is_transient or attempt == max_retries - 1:
                raise
            last_error = e
            wait = base_delay * (2 ** attempt)
            time.sleep(wait)
    raise last_error


def load_dataset(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.DataFrame(json.load(uploaded_file))
    elif name.endswith(".db"):
        return load_sqlite(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}")


def load_sqlite(uploaded_file, table_name: str = None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        conn = sqlite3.connect(tmp_path)
        if table_name is None:
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if tables.empty:
                raise ValueError("No tables found in the .db file")
            table_name = tables["name"].iloc[0]
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    finally:
        os.remove(tmp_path)


def list_sqlite_tables(uploaded_file) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        conn = sqlite3.connect(tmp_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        conn.close()
        return tables["name"].tolist()
    finally:
        os.remove(tmp_path)