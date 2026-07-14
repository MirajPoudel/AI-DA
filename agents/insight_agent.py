from utils import extract_text, invoke_with_retry, split_answer_description
import json

SYSTEM_PROMPT = """You are a business analyst. Given a user's question and the computed result
(which may be a top-10 table), respond with ONLY a JSON object (no markdown fences, no
explanation) with exactly two keys:
- "answer": one direct sentence that answers the user's question using the computed result.
- "description": 2-3 concise plain-English sentences elaborating — notable patterns in the
  data and one practical takeaway.
No headers, no bullet points, no markdown inside the values."""


def generate_insight(user_query: str, result, llm) -> dict:
    prompt = f"User question: {user_query}\n\nComputed result:\n{result}"
    response = invoke_with_retry(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    text = extract_text(response)
    cleaned = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
        answer = str(data.get("answer", "")).strip()
        description = str(data.get("description", "")).strip()
        if answer or description:
            return {"answer": answer, "description": description}
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: model didn't return valid JSON — split the raw text heuristically.
    answer, description = split_answer_description(text)
    return {"answer": answer, "description": description}
