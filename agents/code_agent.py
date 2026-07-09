from langchain_ollama import ChatOllama
import json

llm = ChatOllama(model="phi3", temperature=0)

SYSTEM_PROMPT = """You write Python code for pandas/plotly data analysis.
Rules:
- Assume a DataFrame named `df` already exists.
- Use only pandas, numpy, and plotly.express (as px) / plotly.graph_objects (as go).
- Store the final numeric/table result in a variable named `result`.
- If a chart is needed, store the plotly figure in a variable named `fig`.
- Do NOT read/write files, do NOT use exec/eval/os/sys/subprocess.
- Output ONLY the Python code, no markdown fences, no explanation.
"""


def generate_code(plan: dict, profile: dict, user_query: str) -> str:
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

    code = response.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code
