from utils import extract_text, invoke_with_retry
import json

SYSTEM_PROMPT = """You write Python code for pandas/plotly data analysis.
Rules:
- Assume a DataFrame named `df` already exists.
- Use only pandas, numpy, and plotly.express (as px) / plotly.graph_objects (as go).
- Always compute/aggregate over the FULL dataset (`df`) — never sample or pre-truncate
  before analysing. Only truncate at the very end, for presentation.
- Store the final result in a variable named `result`.
  - If the result is tabular (a DataFrame/Series with more than 10 rows), sort it in the
    most meaningful order for the question (e.g. descending by the key metric) and keep
    only the top 10 rows before assigning it to `result`.
  - If the result is naturally a single scalar/short value, leave it as-is.
- If a chart is needed, store the plotly figure in a variable named `fig`.
  - The chart must plot the SAME records as `result` (e.g. the same top-10 rows) — do not
    chart the full dataset if `result` was truncated to the top 10.
  - Never leave raw column names as axis titles or legend labels. Always pass a `labels={...}`
    dict to plotly express (e.g. `labels={"col_name": "Readable Label"}`) or set
    `fig.update_xaxes(title="...")` / `fig.update_yaxes(title="...")` with human-readable text.
  - Make charts visually polished: use color_discrete_sequence=px.colors.qualitative.Bold
    (categorical) or color_continuous_scale="Viridis" (continuous), set template="plotly_white",
    give it a clear, descriptive title, and apply fig.update_layout(font=dict(size=13),
    legend_title_text="") so legends read cleanly.
- Do NOT read/write files, do NOT use exec/eval/os/sys/subprocess.
- Output ONLY the Python code, no markdown fences, no explanation.
"""


def generate_code(plan: dict, profile: dict, user_query: str, llm) -> str:
    prompt = f"""Dataset columns: {profile['columns']}
Numeric columns: {profile['numeric_columns']}
Categorical columns: {profile['categorical_columns']}

Plan: {json.dumps(plan)}
User question: {user_query}

Write the code now."""

    response = invoke_with_retry(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

    code = extract_text(response)
    code = code.replace("```python", "").replace("```", "").strip()
    return code
