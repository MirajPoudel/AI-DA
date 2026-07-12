# 🧠 AI Data Analyst

A multi-agent data analysis app built with **Streamlit** and **LangGraph**. Upload a CSV, Excel, JSON, or SQLite dataset, ask questions in plain English, and get computed results, interactive charts, and plain-English business insights — then export everything as a styled PDF report.

Supports **Google Gemini** and **OpenAI (ChatGPT)** — paste your API key directly in the UI, no `.env` file needed.

---

## Features

- 📂 Upload CSV, Excel (.xlsx/.xls), JSON, or SQLite (.db) files
- 💬 Ask questions in plain English — the AI plans, codes, and explains
- 📊 Auto-generated interactive Plotly charts (colorful, titled)
- 📄 Export full session as a PDF report (direct answer + description + bordered table + chart image)
- 🔑 Built-in API key selector — choose Gemini or OpenAI, pick a model, paste your key (provider auto-detected from key format)
- 💬 Persistent chat history — sessions saved locally, restore any past conversation from the sidebar
- 🔒 Code sandbox — generated code runs in an isolated subprocess with timeout + banned-pattern checks

---

## Quickstart

### Prerequisites — install Python first

<details>
<summary><strong>🍎 macOS</strong></summary>

**Option A — Homebrew (recommended)**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python
```

**Option B — python.org installer**
Download the macOS installer from https://www.python.org/downloads/ and run it.

Verify:
```bash
python3 --version   # should print Python 3.10 or higher
pip3 --version
```

</details>

<details>
<summary><strong>🪟 Windows</strong></summary>

Download the installer from https://www.python.org/downloads/  
**Important:** tick ✅ *"Add Python to PATH"* before clicking Install.

Verify in Command Prompt or PowerShell:
```powershell
python --version    # should print Python 3.10 or higher
pip --version
```

</details>

---

### 1. Clone the repository

```bash
git clone https://github.com/MirajPoudel/AI-DA.git
cd AI-DA
```

---

### 2. Create a virtual environment

<table>
<tr>
<td><strong>🍎 macOS</strong></td>
<td><strong>🪟 Windows</strong></td>
</tr>
<tr>
<td>

```bash
python3 -m venv venv
source venv/bin/activate
```

</td>
<td>

```powershell
python -m venv venv
venv\Scripts\activate
```

</td>
</tr>
</table>

You should see `(venv)` at the start of your terminal prompt.

---

### 3. Install dependencies

<table>
<tr>
<td><strong>🍎 macOS</strong></td>
<td><strong>🪟 Windows</strong></td>
</tr>
<tr>
<td>

```bash
pip3 install -r requirements.txt
pip3 install fpdf2 "kaleido==0.2.1" langchain-openai
```

</td>
<td>

```powershell
pip install -r requirements.txt
pip install fpdf2 "kaleido==0.2.1" langchain-openai
```

</td>
</tr>
</table>

> **Note:** `kaleido==0.2.1` is required for PDF chart rendering. Version 1.x requires a separate Chrome install and will not work out of the box.

---

### 4. Get an API key

| Provider | Where to get a key | Key format | Free tier |
|---|---|---|---|
| Google Gemini | https://aistudio.google.com/app/apikey | `AIza...` | Yes (limited daily quota) |
| OpenAI (ChatGPT) | https://platform.openai.com/api-keys | `sk-...` | Requires billing credits |

---

### 5. Run the app

<table>
<tr>
<td><strong>🍎 macOS</strong></td>
<td><strong>🪟 Windows</strong></td>
</tr>
<tr>
<td>

```bash
streamlit run app.py
```

</td>
<td>

```powershell
streamlit run app.py
```

</td>
</tr>
</table>

Open your browser at **http://localhost:5000**

---

### 6. Use the app

1. **Sidebar → AI Provider**: paste your API key and click **✅ Apply API Key**  
   *(provider is auto-detected from the key format — `AIza...` → Gemini, `sk-...` → OpenAI)*
2. **Sidebar → Dataset**: upload your CSV / Excel / JSON / SQLite file
3. Type a question in the chat box at the bottom
4. Click **📄 Generate PDF Report** to download a formatted report of your session

---

### Stopping the app

Press `Ctrl + C` in the terminal window where Streamlit is running. To deactivate the virtual environment afterwards:

```bash
deactivate
```

---

## Running on Replit

No local setup needed — open the Repl, dependencies are already installed, and the app runs on port `5000` via the configured workflow (`streamlit run app.py`). Just paste your API key in the sidebar.

---

## Supported Models

**Google Gemini**
- `gemini-2.0-flash` *(fast, recommended — good free tier)*
- `gemini-1.5-pro`
- `gemini-1.5-flash`

**OpenAI**
- `gpt-4o-mini` *(fast, cost-effective — ~$0.15 / 1M tokens)*
- `gpt-4o`
- `gpt-3.5-turbo`

---

## API Key Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `RESOURCE_EXHAUSTED` / `429` on Gemini | Free-tier daily quota hit | Wait until midnight Pacific, or switch to a paid key |
| `insufficient_quota` on OpenAI | No billing credits | Add a card at https://platform.openai.com/settings/billing |
| `invalid_api_key` | Wrong key pasted | Double-check the key — Gemini keys start with `AIza`, OpenAI with `sk-` |

---

## Architecture

```
app.py                  — Streamlit UI, API key selector, session state, chat history
graph.py                — LangGraph pipeline wiring all agents together
sandbox.py              — Isolated subprocess executor with timeout + banned-pattern guard
utils.py                — Dataset loader, LLM retry helper, quota error detection
sessions.py             — Local session persistence (sessions.json)
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

## PDF Report Structure

Each question in the report includes:

1. **Question banner** — blue header with the question text
2. **Direct answer** — first sentence of the insight, bold and large
3. **Description** — 2–3 sentence elaboration, justified
4. **Comparison table** — bordered, styled, alternating row shading (if result is a DataFrame)
5. **Chart image** — full-width Plotly chart rendered as PNG via kaleido

---

## Known Limitations

- Single dataset per session — no multi-file joins
- Sandbox is a locked-down local subprocess, not a remote sandbox — suitable for personal or classroom use, not for untrusted public users
- PDF chart rendering requires `kaleido==0.2.1`; kaleido ≥1.0 requires a system Chrome install
