from langchain_google_genai import ChatGoogleGenerativeAI
from utils import extract_text, invoke_with_retry
import json

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

SYSTEM_PROMPT = """You are a data analysis planner. Given a dataset profile and a user question,
output ONLY a JSON object (no markdown, no explanation) with these keys:
- "operation": one of ["groupby", "filter", "aggregate", "correlation", "trend", "distribution", "comparison"]
- "columns": list of relevant column names from the dataset
- "chart_type": one of ["bar", "line", "pie", "histogram", "scatter", "box", "heatmap", "none"]
- "reasoning": one short sentence explaining the plan
"""


def plan_query(profile: dict, user_query: str) -> dict:
    prompt = f"""Dataset profile:
{json.dumps(profile, default=str)[:3000]}

User question: {user_query}

Respond with JSON only."""

    response = invoke_with_retry(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

    text = extract_text(response)
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"operation": "aggregate", "columns": [], "chart_type": "none",
                "reasoning": "fallback: could not parse plan"}