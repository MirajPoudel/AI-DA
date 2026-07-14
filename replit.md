# AI Data Analyst

## Overview
A multi-agent data analysis app built with Streamlit and LangGraph. Upload a CSV,
Excel, JSON, or SQLite dataset, ask questions in plain English, and get computed
results, interactive Plotly charts, and plain-English business insights — then
export everything as a styled PDF report. Supports Google Gemini, OpenAI, and
Groq (free) — API keys are pasted directly in the sidebar UI, no `.env` needed.

## How it runs on Replit
- Workflow "Start application" runs `streamlit run app.py` on port 5000.
- Dependencies are managed via `requirements.txt` / the Replit Python package
  installer — no manual venv setup needed inside Replit.
- No secrets are required to run the app itself; users provide their own
  Gemini/OpenAI/Groq API key at runtime through the sidebar.

## Architecture
- `app.py` — Streamlit UI: provider/key selection, dataset upload, chat, PDF export.
- `graph.py` — LangGraph pipeline: profile → plan → code → execute → insight.
- `agents/` — the four pipeline agents (profiler, query planner, code generator,
  insight generator).
- `sandbox.py` — runs LLM-generated pandas/plotly code in an isolated subprocess
  with a timeout and banned-pattern checks.
- `auto_analysis.py` — dataset-wide stats/correlation/top-category/distribution
  helpers used for the automatic "Full Dataset Analysis" PDF.
- `pdf_export.py` — builds both the per-question PDF report and the full-dataset
  analysis PDF (fpdf2 + Plotly-to-image via kaleido).
- `sessions.py` — persists chat history to `sessions.json` (DataFrames as JSON,
  figures as `fig.to_json()`).

## Behavior conventions
- Chat answers are structured as: a direct one-line answer, a 2-3 sentence
  description, a "Comparison Table (Top 10)" (full dataset is analyzed, but only
  the top 10 rows are shown, sorted meaningfully), and a chart plotting that same
  top-10 subset. Charts always use human-readable axis/legend labels instead of
  raw column names.
- Uploading a dataset automatically generates a "Full Dataset Analysis" PDF
  (overview stats, numeric summary, correlation heatmap, top-10 category
  breakdowns, distributions) available via a sidebar download button — independent
  of the chat Q&A flow.
- fpdf2's built-in Helvetica font only supports latin-1: avoid characters like
  em-dashes (—) in any text passed to `pdf.cell`/`multi_cell`; use a plain hyphen.

## User preferences
None recorded yet.
