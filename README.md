# AI Data Analyst (v1 - basics-focused)

A lightweight, LangGraph-orchestrated, multi-agent data analysis app built with Streamlit and a local Ollama model (phi3). Upload a CSV/Excel/JSON dataset, ask questions in plain English, get computed results, a chart, and a short business insight.

## Setup

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
ollama pull phi3
```

## Run

```
python -m streamlit run app.py
```

## Architecture

- `agents/profiler.py` — pandas-only dataset profiling (schema, nulls, dupes, correlations)
- `agents/query_agent.py` — LLM plans the operation/columns/chart type
- `agents/code_agent.py` — LLM generates pandas/plotly code
- `sandbox.py` — runs generated code in an isolated subprocess with a timeout and a banned-pattern check (no os/sys/subprocess/eval/exec/open)
- `agents/insight_agent.py` — LLM turns the result into a short plain-English insight
- `graph.py` — LangGraph wiring the above into one pipeline
- `app.py` — Streamlit chat UI

## Known limitations (by design, for v1)

- Single dataset at a time, no multi-file comparison
- No PostgreSQL/MySQL/SQLite sources yet — CSV/Excel/JSON only
- No PDF report generation yet
- No authentication or persistent history across sessions
- Sandbox is a locked-down local subprocess, not a true remote sandbox (E2B) — fine for personal/class use, not for hosting with untrusted public users
