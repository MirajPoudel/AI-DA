# 🧠 AI Data Analyst

A multi-agent data analysis app built with **Streamlit** and **LangGraph**. Upload a CSV, Excel, JSON, or SQLite dataset, ask questions in plain English, and get computed results, interactive charts, and plain-English business insights — then export everything as a styled PDF report.

Supports **Google Gemini** and **OpenAI (ChatGPT)** — users bring their own API key directly in the UI, so the app works for anyone without touching environment variables.

---

## Features

- 📂 Upload CSV, Excel (.xlsx/.xls), JSON, or SQLite (.db) files
- 💬 Ask questions in plain English — the AI plans, codes, and explains
- 📊 Auto-generated interactive Plotly charts (colorful, titled)
- 📄 Export full session as a PDF report (direct answer + description + bordered table + chart image)
- 🔑 Built-in API key selector — choose Gemini or OpenAI, pick a model, paste your key
- 🔒 Code sandbox — generated code runs in an isolated subprocess with banned-pattern checks

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get an API key

| Provider | Where to get a key | Key format |
|---|---|---|
| Google Gemini | https://aistudio.google.com/app/apikey | `AIza...` |
| OpenAI (ChatGPT) | https://platform.openai.com/api-keys | `sk-...` |

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Use the app

1. Open the app in your browser (default: http://localhost:5000)
2. In the sidebar → **AI Provider**: select your provider, pick a model, paste your API key, click **✅ Apply API Key**
3. In the sidebar → **Dataset**: upload your file
4. Ask questions in the chat box at the bottom
5. Click **📄 Generate PDF Report** after running analyses to download a formatted report

---

## Running on Replit

No extra setup needed — dependencies are installed via `pip install -r requirements.txt` and the app runs on port `5000` via the configured workflow (`streamlit run app.py`).

---

## Supported Models

**Google Gemini**
- `gemini-2.0-flash` *(fast, recommended)*
- `gemini-1.5-pro`
- `gemini-1.5-flash`

**OpenAI**
- `gpt-4o-mini` *(fast, cost-effective)*
- `gpt-4o`
- `gpt-3.5-turbo`

---

## Architecture

```
app.py                  — Streamlit UI, API key selector, session state
graph.py                — LangGraph pipeline wiring all agents together
sandbox.py              — Isolated subprocess executor with timeout + banned-pattern guard
utils.py                — Dataset loader, LLM helper utilities
pdf_export.py           — PDF report generator (fpdf2 + kaleido for chart images)

agents/
  profiler.py           — Pandas-only dataset profiling (schema, nulls, dupes, correlations)
  query_agent.py        — LLM plans the operation, columns, and chart type
  code_agent.py         — LLM generates pandas/plotly analysis code
  insight_agent.py      — LLM writes a plain-English business insight from the result
```

### Pipeline flow

```
User question
     │
     ▼
 [profiler]  →  [query_agent]  →  [code_agent]  →  [sandbox executor]  →  [insight_agent]
                                                              │
                                               result + fig + insight
                                                              │
                                                       Streamlit UI
```

---

## PDF Report structure

Each question in the report includes:

1. **Question banner** (blue header)
2. **Direct answer** — first sentence, bold and large (e.g. *"The West region has the most orders with 83 orders."*)
3. **Description** — 2–3 sentence elaboration
4. **Comparison table** — bordered, styled, with alternating row shading (if result is a DataFrame)
5. **Chart image** — full-width Plotly chart rendered as PNG via kaleido

---

## Known Limitations

- Single dataset per session — no multi-file joins
- No persistent history across browser sessions
- Sandbox is a locked-down local subprocess, not a remote sandbox (E2B) — suitable for personal or classroom use, not for untrusted public users
- PDF chart rendering requires `kaleido==0.2.1` (bundled Chromium); kaleido ≥1.0 requires a system Chrome install
