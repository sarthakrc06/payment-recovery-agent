"""
LLM triage step: for failed-payment events the rule-based + fuzzy classifier
could NOT confidently categorize (classification_method == "unresolved"),
ask Gemini to reason about the case using full context and return a
category + confidence + one-sentence explanation.

This is deliberately the LAST resort, not the first pass -- keeps LLM calls
few, bounded, and auditable (every call and its reasoning is logged).

Usage:
    python src/llm_triage.py
Requires GEMINI_API_KEY in a .env file at the project root.
"""

import os
import csv
import json
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

VALID_CATEGORIES = [
    "insufficient_funds",
    "expired_card",
    "otp_session_timeout",
    "gateway_method_degraded",
    "risk_block",
]

PROMPT_TEMPLATE = """You are a payments risk analyst. A payment failed and could not be
automatically classified by rule-based checks. Based on the details below, decide which
ONE root-cause category best explains it.

Categories (pick exactly one):
- insufficient_funds: the payer didn't have enough balance
- expired_card: the card used is expired or invalid
- otp_session_timeout: the payer's OTP entry or session timed out
- gateway_method_degraded: a technical/bank-side/gateway issue, not the payer's fault
- risk_block: blocked by fraud/risk systems, needs manual review

Transaction details:
- Payment method: {payment_method}
- Amount: Rs {amount}
- Error code: {error_code}
- Error message: {error_message}

Respond with ONLY a JSON object, no markdown formatting, no code fences, in exactly
this shape:
{{"category": "<one of the five category names above>", "confidence": <0-100 integer>,
"reasoning": "<one sentence explaining why>"}}
"""


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Make sure it's set in a .env file at the project root."
        )
    return genai.Client(api_key=api_key)


def parse_llm_response(text: str):
    """Strips markdown code fences if present, then parses JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    category = data.get("category")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Model returned an invalid category: {category!r}")

    return {
        "category": category,
        "confidence": int(data.get("confidence", 0)),
        "reasoning": data.get("reasoning", ""),
    }


def triage_event(client, row):
    prompt = PROMPT_TEMPLATE.format(
        payment_method=row["payment_method"],
        amount=row["amount_inr"],
        error_code=row["error_code"],
        error_message=row["error_message"],
    )
    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return parse_llm_response(response.text)


def run_llm_triage(classified_path="data/classified.csv", output_path="data/classified.csv"):
    with open(classified_path, newline="") as f:
        rows = list(csv.DictReader(f))

    client = get_client()
    triaged_count = 0

    for row in rows:
        if row["classification_method"] != "unresolved":
            continue

        try:
            result = triage_event(client, row)
            row["predicted_root_cause"] = result["category"]
            row["confidence"] = result["confidence"]
            row["classification_method"] = "llm_triage"
            row["llm_reasoning"] = result["reasoning"]
            triaged_count += 1
            print(f"  {row['transaction_id']}: -> {result['category']} "
                  f"({result['confidence']}%) -- {result['reasoning']}")
        except Exception as e:
            row["llm_reasoning"] = f"LLM triage failed: {e}"
            print(f"  {row['transaction_id']}: LLM triage failed ({e}), staying 'unknown'")

        time.sleep(0.5)  # gentle on free-tier rate limits

    if "llm_reasoning" not in rows[0]:
        for row in rows:
            row.setdefault("llm_reasoning", "")

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLLM triage complete: {triaged_count} case(s) resolved.")
    print(f"  -> {output_path}")


if __name__ == "__main__":
    run_llm_triage()
