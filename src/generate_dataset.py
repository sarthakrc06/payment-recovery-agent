"""
Generates synthetic failed-payment events for the Payment Failure Recovery Agent.

Each event has:
  - transaction details (amount, method, timestamp)
  - an error_code and error_message (mimicking real gateway failure reasons)
  - a hidden ground-truth root_cause_category (what the classifier SHOULD say)
  - a hidden ground-truth recoverable flag + recovery_probability
    (used later to simulate whether a recovery action would actually succeed)

The ground truth is written to a SEPARATE file (ground_truth.csv) so the
classifier/agent never sees it directly -- it only sees the "public" columns,
exactly like a real held-out test set.
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)  # reproducible dataset

# ---- Root cause categories and how they manifest ----
# Each category has: typical error codes/messages, a base recovery probability
# (if the RIGHT action is taken), and a "wrong action" penalty (if a generic
# retry is used instead of the right one).

CATEGORIES = {
    "insufficient_funds": {
        "codes": ["BAD_REQUEST_ERROR_INSUFFICIENT_FUNDS", "GATEWAY_ERROR_INSUFFICIENT_BALANCE"],
        "messages": [
            "Payment failed due to insufficient balance in account",
            "Transaction declined: insufficient funds",
        ],
        "methods": ["card", "netbanking"],
        "right_action": "delayed_retry_or_switch_method",
        "recovery_prob_right": 0.55,
        "recovery_prob_generic": 0.15,
    },
    "expired_card": {
        "codes": ["CARD_EXPIRED", "GATEWAY_ERROR_EXPIRED_CARD"],
        "messages": [
            "Card has expired",
            "Transaction declined: card expiry date invalid",
        ],
        "methods": ["card"],
        "right_action": "prompt_update_card_or_switch_method",
        "recovery_prob_right": 0.65,
        "recovery_prob_generic": 0.05,
    },
    "otp_session_timeout": {
        "codes": ["OTP_TIMEOUT", "SESSION_EXPIRED"],
        "messages": [
            "OTP entry timed out",
            "Payment session expired before completion",
        ],
        "methods": ["card", "upi", "netbanking"],
        "right_action": "immediate_retry_fresh_session",
        "recovery_prob_right": 0.80,
        "recovery_prob_generic": 0.55,
    },
    "gateway_method_degraded": {
        "codes": ["GATEWAY_TIMEOUT", "BANK_SERVER_ERROR"],
        "messages": [
            "Bank server not responding",
            "Gateway timeout while processing payment",
        ],
        "methods": ["upi", "netbanking"],
        "right_action": "auto_switch_method",
        "recovery_prob_right": 0.60,
        "recovery_prob_generic": 0.20,
    },
    "risk_block": {
        "codes": ["RISK_ENGINE_BLOCKED", "FRAUD_SUSPECTED"],
        "messages": [
            "Transaction blocked by risk engine",
            "Payment flagged for manual review",
        ],
        "methods": ["card", "upi"],
        "right_action": "manual_review_no_retry",
        "recovery_prob_right": 0.30,   # only recovers if manual review clears it
        "recovery_prob_generic": 0.02,  # retrying a risk-block usually just fails again
    },
}

METHOD_LABELS = {
    "card": "Card",
    "upi": "UPI",
    "netbanking": "Netbanking",
}


def random_timestamp():
    start = datetime(2026, 6, 1)
    offset_minutes = random.randint(0, 60 * 24 * 60)  # within ~60 days
    return start + timedelta(minutes=offset_minutes)


# Noisy variants: same real-world cause, but with an error code our rule
# table has NEVER seen (forces the fuzzy-message fallback to kick in),
# using a slightly reworded message (tests fuzzy matching, not exact match).
NOISY_MESSAGE_VARIANTS = {
    "insufficient_funds": "Payment could not be completed - not enough balance available",
    "expired_card": "Your card's validity period has ended",
    "otp_session_timeout": "The one-time password window closed before verification",
    "gateway_method_degraded": "Unable to reach the bank's servers at this time",
    "risk_block": "This payment was stopped by our fraud prevention checks",
}

# Genuinely ambiguous events: neither code nor message maps cleanly to any
# category. These SHOULD end up as "unknown" -> routed to LLM triage.
AMBIGUOUS_EVENTS = [
    {"code": "PROCESSING_ERROR_9921", "message": "Payment could not be processed at this time"},
    {"code": "UNKNOWN_DECLINE", "message": "Transaction was not approved"},
    {"code": "GATEWAY_ERR_MISC", "message": "Something went wrong, please contact support"},
]


def generate_events(n=70, noisy_count=10, ambiguous_count=3):
    events = []
    ground_truth = []

    # Reserve some indices near the end for noisy + ambiguous cases
    n_clean = n - noisy_count - ambiguous_count

    for i in range(1, n_clean + 1):
        category = random.choice(list(CATEGORIES.keys()))
        spec = CATEGORIES[category]

        code = random.choice(spec["codes"])
        message = random.choice(spec["messages"])
        method = random.choice(spec["methods"])
        amount = round(random.uniform(150, 25000), 2)
        ts = random_timestamp()

        txn_id = f"pay_{100000 + i}"

        events.append({
            "transaction_id": txn_id,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_inr": amount,
            "payment_method": METHOD_LABELS[method],
            "error_code": code,
            "error_message": message,
        })

        ground_truth.append({
            "transaction_id": txn_id,
            "true_root_cause": category,
            "right_action": spec["right_action"],
            "recovery_prob_if_right_action": spec["recovery_prob_right"],
            "recovery_prob_if_generic_retry": spec["recovery_prob_generic"],
        })

    # ---- Noisy cases: unrecognized code, reworded message (fuzzy fallback test) ----
    for j in range(noisy_count):
        category = random.choice(list(CATEGORIES.keys()))
        spec = CATEGORIES[category]
        method = random.choice(spec["methods"])
        amount = round(random.uniform(150, 25000), 2)
        ts = random_timestamp()
        txn_id = f"pay_{100000 + n_clean + j + 1}"

        events.append({
            "transaction_id": txn_id,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_inr": amount,
            "payment_method": METHOD_LABELS[method],
            "error_code": f"UNRECOGNIZED_CODE_{random.randint(1000,9999)}",
            "error_message": NOISY_MESSAGE_VARIANTS[category],
        })
        ground_truth.append({
            "transaction_id": txn_id,
            "true_root_cause": category,
            "right_action": spec["right_action"],
            "recovery_prob_if_right_action": spec["recovery_prob_right"],
            "recovery_prob_if_generic_retry": spec["recovery_prob_generic"],
        })

    # ---- Genuinely ambiguous cases: should end up "unknown" -> LLM triage ----
    for k in range(ambiguous_count):
        amb = AMBIGUOUS_EVENTS[k % len(AMBIGUOUS_EVENTS)]
        # Assign a real underlying cause (for scoring honesty) even though
        # it's not recoverable from the code/message alone -- this is the
        # realistic case an LLM triage step is meant to catch via context.
        category = random.choice(list(CATEGORIES.keys()))
        spec = CATEGORIES[category]
        method = random.choice(spec["methods"])
        amount = round(random.uniform(150, 25000), 2)
        ts = random_timestamp()
        txn_id = f"pay_{100000 + n_clean + noisy_count + k + 1}"

        events.append({
            "transaction_id": txn_id,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_inr": amount,
            "payment_method": METHOD_LABELS[method],
            "error_code": amb["code"],
            "error_message": amb["message"],
        })
        ground_truth.append({
            "transaction_id": txn_id,
            "true_root_cause": category,
            "right_action": spec["right_action"],
            "recovery_prob_if_right_action": spec["recovery_prob_right"],
            "recovery_prob_if_generic_retry": spec["recovery_prob_generic"],
        })

    return events, ground_truth


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    events, ground_truth = generate_events(n=70)

    write_csv(
        events,
        "data/failed_payments.csv",
        fieldnames=["transaction_id", "timestamp", "amount_inr", "payment_method", "error_code", "error_message"],
    )
    write_csv(
        ground_truth,
        "data/ground_truth.csv",
        fieldnames=["transaction_id", "true_root_cause", "right_action", "recovery_prob_if_right_action", "recovery_prob_if_generic_retry"],
    )

    print(f"Generated {len(events)} failed payment events.")
    print("  -> data/failed_payments.csv   (the 'public' dataset your agent sees)")
    print("  -> data/ground_truth.csv      (used ONLY for scoring, not shown to the classifier)")
