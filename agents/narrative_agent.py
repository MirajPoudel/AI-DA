from utils import extract_text, invoke_with_retry
import json

SYSTEM_PROMPT = """You are a senior data analyst writing a dataset analysis report.
Given structured facts about a dataset (as JSON), respond with ONLY a JSON object
(no markdown fences, no explanation) with exactly these keys:
- "overview": 3-5 sentence paragraph describing what the dataset contains.
- "quality": 3-5 sentence paragraph on data quality (missing values, duplicates,
  suspicious zeros) and what they mean for analysis.
- "key_findings": a list of 4-6 short, punchy bullet-point strings (no leading dashes)
  summarizing the most notable patterns in the facts.
- "summary": 3-4 sentence closing paragraph tying the findings together.
Plain prose only, no markdown formatting, no headers inside the values."""


def generate_report_narrative(facts: dict, llm) -> dict:
    prompt = f"Dataset facts (JSON):\n{json.dumps(facts, default=str)[:4000]}\n\nWrite the report sections now."
    response = invoke_with_retry(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    text = extract_text(response)
    cleaned = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)

    findings = data.get("key_findings", [])
    if not isinstance(findings, list):
        findings = []

    return {
        "overview": str(data.get("overview", "")).strip(),
        "quality": str(data.get("quality", "")).strip(),
        "key_findings": [str(x).strip() for x in findings if str(x).strip()],
        "summary": str(data.get("summary", "")).strip(),
    }
