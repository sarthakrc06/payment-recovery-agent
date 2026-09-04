"""
For each classified failure, picks the right recovery action, simulates
whether it succeeds, and produces the final "money recovered" report.

Key design point: the agent only sees its PREDICTED root cause and picks
an action based on that (like a real system would). Whether the action
actually succeeds is simulated using the TRUE root cause's probabilities
(from ground_truth.csv) -- reality doesn't care what the model believed.
This means a misclassification can genuinely lead to a worse outcome,
exactly like it would in production. ground_truth.csv is used ONLY here,
for simulating the real-world result -- never as an input to the decision.

Stopping rule: at most ONE retry attempt per transaction. Risk-block
predictions never get an automatic retry -- they're escalated to manual
review instead (a smaller, separate "manual recovery" chance applies).
"""

import csv
import random

# ---- Action to take, based on the PREDICTED category ----
ACTION_FOR_CATEGORY = {
    "insufficient_funds": "delayed_retry_or_switch_method",
    "expired_card": "prompt_update_card_or_switch_method",
    "otp_session_timeout": "immediate_retry_fresh_session",
    "gateway_method_degraded": "auto_switch_method",
    "risk_block": "manual_review_no_retry",
    "unknown": "manual_review_no_retry",  # safe default: don't guess-retry
}

MANUAL_REVIEW_RECOVERY_PROB = 0.20  # chance an escalated case gets cleared manually


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_recovery(classified_path="data/classified.csv", ground_truth_path="data/ground_truth.csv",
                  output_path="data/recovery_results.csv"):
    random.seed(7)  # reset every call -> same simulated outcome every run, not just the first

    classified = {r["transaction_id"]: r for r in load_csv(classified_path)}
    ground_truth = {r["transaction_id"]: r for r in load_csv(ground_truth_path)}

    results = []

    for txn_id, event in classified.items():
        truth = ground_truth[txn_id]
        predicted_category = event["predicted_root_cause"]
        true_category = truth["true_root_cause"]
        amount = float(event["amount_inr"])

        action = ACTION_FOR_CATEGORY[predicted_category]

        if action == "manual_review_no_retry":
            # No automatic retry attempted -- escalated instead.
            success = random.random() < MANUAL_REVIEW_RECOVERY_PROB
            outcome = "recovered_via_manual_review" if success else "escalated_unresolved"
            reasoning = (
                f"Predicted '{predicted_category}' -> no auto-retry (stopping rule: "
                f"risk/unknown cases are never auto-retried). Escalated to manual review."
            )
        else:
            # Did the chosen action match what would ACTUALLY be right for this case?
            right_action_for_true_cause = ACTION_FOR_CATEGORY[true_category]
            if action == right_action_for_true_cause:
                prob = float(truth["recovery_prob_if_right_action"])
                reasoning = f"Predicted '{predicted_category}' (correct) -> right action taken."
            else:
                prob = float(truth["recovery_prob_if_generic_retry"])
                reasoning = (
                    f"Predicted '{predicted_category}' but true cause was '{true_category}' "
                    f"-> mismatched action, lower success odds."
                )

            success = random.random() < prob
            outcome = "recovered" if success else "not_recovered"

        results.append({
            "transaction_id": txn_id,
            "amount_inr": amount,
            "predicted_root_cause": predicted_category,
            "true_root_cause": true_category,  # kept for audit/debug transparency
            "classification_correct": predicted_category == true_category,
            "action_taken": action,
            "outcome": outcome,
            "amount_recovered": amount if "recovered" in outcome else 0.0,
            "reasoning": reasoning,
        })

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results


def print_report(results):
    total_amount = sum(r["amount_inr"] for r in results)
    recovered_amount = sum(r["amount_recovered"] for r in results)
    recovered_count = sum(1 for r in results if r["amount_recovered"] > 0)

    print(f"\n=== Recovery Report ===")
    print(f"Total transactions:     {len(results)}")
    print(f"Total amount at risk:   Rs {total_amount:,.2f}")
    print(f"Amount recovered:       Rs {recovered_amount:,.2f}")
    print(f"Overall recovery rate:  {recovered_amount/total_amount*100:.1f}% of amount "
          f"({recovered_count}/{len(results)} transactions)")

    print(f"\n--- By predicted category ---")
    categories = sorted(set(r["predicted_root_cause"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["predicted_root_cause"] == cat]
        cat_total = sum(r["amount_inr"] for r in cat_results)
        cat_recovered = sum(r["amount_recovered"] for r in cat_results)
        rate = (cat_recovered / cat_total * 100) if cat_total else 0
        print(f"  {cat:28s} n={len(cat_results):3d}  recovered Rs {cat_recovered:>10,.2f} "
              f"/ Rs {cat_total:>10,.2f}  ({rate:.1f}%)")

    print(f"\n--- Honest exception list (not recovered) ---")
    unresolved = [r for r in results if r["amount_recovered"] == 0.0]
    for r in unresolved:
        print(f"  {r['transaction_id']}: Rs {r['amount_inr']:.2f} | {r['outcome']} | {r['reasoning']}")
    print(f"  Total unresolved: {len(unresolved)} transactions, "
          f"Rs {sum(r['amount_inr'] for r in unresolved):,.2f} still at risk")


if __name__ == "__main__":
    results = run_recovery()
    print_report(results)
    print(f"\n  -> data/recovery_results.csv")
