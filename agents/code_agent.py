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
  - If the category axis would show long or numerous text values (e.g. full names, titles,
    dates, sentences — anything long, not short codes like country abbreviations), do NOT spell
    them out across the chart. Instead add a leading rank/position column (e.g. "#": 1, 2, 3...)
    to `result`, plot that short column on the axis instead of the raw text, and pass the full
    text via `hover_data={"original_col": True}` so it only appears in the table (`result`,
    already shown to the user) and in hover tooltips — never crowding the chart itself.
  - If a categorical column packs multiple values into one string using a separator (e.g. a
    movie's genres stored as "Action|Adventure|Comedy", using "|", ";", or "/"), split that
    column and explode it into one row per individual value BEFORE counting/grouping/charting
    (`df.assign(col=df["col"].str.split("|")).explode("col")`, stripping whitespace). A record
    then belongs under each of its individual values (e.g. once under "Action" and once under
    "Adventure"), never under the combined string as if it were a single category.
  - Make charts visually polished: use color_discrete_sequence=px.colors.qualitative.Bold
    (categorical) or color_continuous_scale="Viridis" (continuous), set template="plotly_white",
    give it a clear, descriptive title, and apply fig.update_layout(font=dict(size=13),
    legend_title_text="", showlegend=False if color already encodes the position column) so
    legends read cleanly.
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
