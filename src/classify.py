"""
Classifies each failed payment event into a root-cause category.

Two-pass approach:
  1. RULE-BASED (fast, high-confidence): exact match on error_code against
     a known lookup table. Handles the clear-cut majority of cases.
  2. FUZZY FALLBACK: for any error_code we don't recognize, fuzzy-match the
     error_message text against known category message patterns. Catches
     messier/unseen real-world variants.
  3. Anything still below a confidence threshold is marked "unknown" --
     these are exactly the cases that should go to the LLM triage step.

Output: data/classified.csv with predicted_root_cause, confidence, and
        the method used (rule / fuzzy / unknown) for each transaction.
"""

import csv
from rapidfuzz import fuzz

# ---- Known error codes per category (rule-based, high confidence) ----
CODE_TO_CATEGORY = {
    "BAD_REQUEST_ERROR_INSUFFICIENT_FUNDS": "insufficient_funds",
    "GATEWAY_ERROR_INSUFFICIENT_BALANCE": "insufficient_funds",
    "CARD_EXPIRED": "expired_card",
    "GATEWAY_ERROR_EXPIRED_CARD": "expired_card",
    "OTP_TIMEOUT": "otp_session_timeout",
    "SESSION_EXPIRED": "otp_session_timeout",
    "GATEWAY_TIMEOUT": "gateway_method_degraded",
    "BANK_SERVER_ERROR": "gateway_method_degraded",
    "RISK_ENGINE_BLOCKED": "risk_block",
    "FRAUD_SUSPECTED": "risk_block",
}

# ---- Keyword signals per category (fuzzy fallback) ----
# Full-sentence fuzzy matching (token_sort_ratio) turned out to score
# genuine paraphrases too low (~30-50), since a reworded sentence shares
# few characters/word-order with the canned pattern even when it means
# the same thing. Keyword-level fuzzy matching is more robust: we check
# whether words *semantically core* to each category appear anywhere in
# the message, fuzzy-matched individually (catches typos/inflections too).
CATEGORY_KEYWORDS = {
    "insufficient_funds": ["balance", "funds", "insufficient", "enough"],
    "expired_card": ["expired", "expiry", "validity"],
    "otp_session_timeout": ["otp", "session", "timed out", "timeout", "window", "password"],
    "gateway_method_degraded": ["gateway", "bank server", "servers", "timeout", "responding", "reach"],
    "risk_block": ["risk", "fraud", "blocked", "flagged", "review", "suspicious"],
}

FUZZY_CONFIDENCE_THRESHOLD = 60  # 0-100 scale; below this -> "unknown"


def classify_by_code(error_code: str):
    """Pass 1: exact lookup. Returns (category, confidence) or (None, 0)."""
    category = CODE_TO_CATEGORY.get(error_code)
    if category:
        return category, 100
    return None, 0


def classify_by_message(error_message: str):
    """
    Pass 2: keyword-level fuzzy match. For each category, check whether any
    of its core keywords appear (fuzzily, to tolerate typos/variants)
    anywhere in the message. Score = best keyword match found, per category.
    """
    message = error_message.lower()
    words_in_message = message.split()

    best_category = None
    best_score = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        category_best = 0
        for keyword in keywords:
            if " " in keyword:
                # multi-word keyword: check as substring with partial_ratio
                score = fuzz.partial_ratio(keyword, message)
            else:
                # single word: best fuzzy match against any word in the message
                score = max((fuzz.ratio(keyword, w) for w in words_in_message), default=0)
            category_best = max(category_best, score)

        if category_best > best_score:
            best_score = category_best
            best_category = category

    if best_score >= FUZZY_CONFIDENCE_THRESHOLD:
        return best_category, best_score
    return None, best_score


def classify_event(error_code: str, error_message: str):
    """Runs the two-pass classification for a single event."""
    category, confidence = classify_by_code(error_code)
    if category:
        return category, confidence, "rule"

    category, confidence = classify_by_message(error_message)
    if category:
        return category, confidence, "fuzzy"

    return "unknown", confidence, "unresolved"


def classify_dataset(input_path="data/failed_payments.csv", output_path="data/classified.csv"):
    rows = []
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category, confidence, method = classify_event(row["error_code"], row["error_message"])
            row["predicted_root_cause"] = category
            row["confidence"] = confidence
            row["classification_method"] = method
            rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    rows = classify_dataset()

    method_counts = {}
    for row in rows:
        m = row["classification_method"]
        method_counts[m] = method_counts.get(m, 0) + 1

    print(f"Classified {len(rows)} events.")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")
    print("  -> data/classified.csv")
