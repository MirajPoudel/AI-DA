from utils import extract_text

SYSTEM_PROMPT = """You are a business analyst. Given a user's question and the computed result,
write 2-4 concise sentences of plain-English insight: what the numbers show, any notable
pattern, and one practical takeaway. No headers, no bullet points, no markdown."""


def generate_insight(user_query: str, result, llm) -> str:
    prompt = f"User question: {user_query}\n\nComputed result:\n{result}"
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return extract_text(response)
