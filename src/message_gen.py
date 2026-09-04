"""
Drafts a short, customer-facing recovery message for every transaction where
the agent decided to actually retry (not the manual-review/risk-block cases,
which get an internal escalation note instead).

Deliberately BATCHED into a single Gemini call (not one call per transaction)
-- same "bounded, cheap, auditable" principle as llm_triage.py, just applied
to generation instead of classification. One call drafts messages for the
whole batch instead of dozens of separate calls.

Usage:
    python src/message_gen.py
Requires GEMINI_API_KEY in a .env file at the project root.
"""

import os
import csv
import json

from dotenv import load_dotenv
from google import genai

load_dotenv()

BATCH_PROMPT_TEMPLATE = """You are drafting short SMS/notification messages for a payments app.
For each failed transaction below, write a brief, friendly, action-oriented message (max 25 words)
that would nudge the customer to complete their payment. Match the tone to the failure reason --
e.g. insufficient_funds should be gentle and suggest an alternative, otp_session_timeout should
be urgent and simple, expired_card should prompt updating payment details.

Do NOT mention internal system details like error codes. Write as if speaking directly to the
customer. Use Rs for the currency symbol.

Transactions:
{transactions_json}

Respond with ONLY a JSON array, no markdown formatting, no code fences, in exactly this shape:
[{{"transaction_id": "...", "message": "..."}}, ...]
One entry per transaction, same order as given, same transaction_id values.
"""

# These categories get an internal note instead of a customer message --
# we don't want to prompt a retry on something the risk engine flagged.
NO_CUSTOMER_MESSAGE_CATEGORIES = {"risk_block"}


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Make sure it's set in a .env file at the project root."
        )
    return genai.Client(api_key=api_key)


def parse_batch_response(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def generate_messages(results_path="data/recovery_results.csv",
                       output_path="data/recovery_messages.csv"):
    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))

    actionable = [
        r for r in rows
        if r["predicted_root_cause"] not in NO_CUSTOMER_MESSAGE_CATEGORIES
    ]

    if not actionable:
        print("No actionable transactions to draft messages for.")
        return []

    batch_input = [
        {
            "transaction_id": r["transaction_id"],
            "predicted_root_cause": r["predicted_root_cause"],
            "amount_inr": r["amount_inr"],
        }
        for r in actionable
    ]

    client = get_client()
    prompt = BATCH_PROMPT_TEMPLATE.format(transactions_json=json.dumps(batch_input, indent=2))

    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    messages = parse_batch_response(response.text)

    message_by_id = {m["transaction_id"]: m["message"] for m in messages}

    output_rows = []
    for r in rows:
        output_rows.append({
            "transaction_id": r["transaction_id"],
            "predicted_root_cause": r["predicted_root_cause"],
            "customer_message": message_by_id.get(
                r["transaction_id"],
                "(internal escalation -- no customer message sent)"
                if r["predicted_root_cause"] in NO_CUSTOMER_MESSAGE_CATEGORIES else ""
            ),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transaction_id", "predicted_root_cause", "customer_message"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Drafted {len(messages)} customer messages in a single batched call.")
    print(f"  -> {output_path}")
    return output_rows


if __name__ == "__main__":
    rows = generate_messages()
    for r in rows[:5]:
        print(f"  {r['transaction_id']} ({r['predicted_root_cause']}): {r['customer_message']}")
