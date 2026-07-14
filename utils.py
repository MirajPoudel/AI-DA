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


def split_answer_description(text: str) -> tuple[str, str]:
    """Fallback split of a plain-text insight into a short direct answer and a
    longer description, used when structured JSON parsing fails."""
    if not text:
        return "", ""
    for sep in (". ", ".\n", "\n\n"):
        idx = text.find(sep)
        if idx != -1 and idx < 200:
            return text[: idx + 1].strip(), text[idx + len(sep):].strip()
    return text.strip(), ""


class QuotaExhaustedError(Exception):
    """Raised when the daily/project quota is permanently exhausted."""
    pass


def invoke_with_retry(llm, messages, max_retries: int = 4, base_delay: float = 5.0):
    """Retries an LLM call on transient rate-limit errors with exponential backoff.

    - 429 per-minute limit  → waits and retries (up to max_retries times)
    - 429 daily quota       → raises QuotaExhaustedError immediately (no point retrying)
    - 503 / UNAVAILABLE     → waits and retries
    - anything else         → re-raises immediately
    """
    DAILY_QUOTA_SIGNALS = (
        "PerDay",
        "daily",
        "free_tier_requests",
        "FreeTier",
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)

            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            is_unavailable = "503" in error_str or "UNAVAILABLE" in error_str

            if is_rate_limit:
                # OpenAI: no billing / trial expired
                if "insufficient_quota" in error_str:
                    raise QuotaExhaustedError(
                        "Your OpenAI account has no remaining credits. "
                        "Add a payment method and buy credits at "
                        "https://platform.openai.com/settings/billing — "
                        "$5 will last a long time with gpt-4o-mini."
                    ) from e
                # If ANY daily-quota signal appears, no amount of waiting helps.
                if any(sig in error_str for sig in DAILY_QUOTA_SIGNALS):
                    raise QuotaExhaustedError(
                        "Your free-tier daily quota is exhausted. "
                        "Try again tomorrow, switch to a paid API key, "
                        "or use a different model."
                    ) from e
                # Per-minute rate limit — back off and retry
                if attempt == max_retries - 1:
                    raise
                last_error = e
                wait = base_delay * (2 ** attempt)   # 5 s, 10 s, 20 s, 40 s
                time.sleep(wait)
                continue

            if is_unavailable:
                if attempt == max_retries - 1:
                    raise
                last_error = e
                wait = base_delay * (2 ** attempt)
                time.sleep(wait)
                continue

            # Any other error — raise immediately
            raise

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