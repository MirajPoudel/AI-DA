from utils import extract_text
import json

SYSTEM_PROMPT = """You write Python code for pandas/plotly data analysis.
Rules:
- Assume a DataFrame named `df` already exists.
- Use only pandas, numpy, and plotly.express (as px) / plotly.graph_objects (as go).
- Store the final numeric/table result in a variable named `result`.
- If a chart is needed, store the plotly figure in a variable named `fig`.
- Always make charts visually rich: use color_discrete_sequence=px.colors.qualitative.Bold
  or color_continuous_scale="Viridis" where appropriate, and set a descriptive title.
- Apply fig.update_layout(template="plotly", font=dict(size=13)) to every figure.
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

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

    code = extract_text(response)
    code = code.replace("```python", "").replace("```", "").strip()
    return code
