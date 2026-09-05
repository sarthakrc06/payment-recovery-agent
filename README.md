# Payment Failure Recovery Agent

**Razorpay AI Buildathon 2026 — Track 03: AI Revenue Recovery**

An agent that diagnoses *why* a payment failed, picks a targeted recovery action for that specific cause, and reports the measured recovery rate — with an honest list of what it couldn't fix.

---

## The problem

When a payment fails, most systems respond the same way to every failure — a generic retry, or nothing at all. But a "failed payment" isn't one problem; it's several different problems wearing the same label:

- The customer didn't have enough balance
- Their card expired
- They fumbled the OTP or their session timed out
- The bank's gateway had a technical hiccup — nothing to do with the customer
- The risk engine flagged it as suspicious

Each of these needs a **different response** to actually recover the money. Treating them identically means either under-reacting (losing a sale a simple nudge would've saved) or over-reacting (auto-retrying a risk-blocked payment, which looks like fraud). Since Razorpay earns a fee on every successful transaction, every recoverable payment that goes unrecovered is lost revenue for the merchant **and** for Razorpay.

## What this does

Given a batch of failed payment events, the agent:

1. **Classifies** each failure into a root-cause category
2. **Selects** the recovery action suited to that specific cause — including correctly choosing *not* to auto-retry risk-blocked cases
3. **Simulates** whether the action would succeed, and computes real recovery metrics
4. **Drafts** the actual customer-facing recovery message for actionable cases (via Gemini)
5. **Reports** everything honestly: recovery rate by category, plus every unresolved case and why

## Architecture

```
Failed payment event (error code, message, amount, method)
        │
        ▼
Classify root cause  ──┬── Pass 1: exact error-code lookup      (rule-based, ~100% confidence)
                        ├── Pass 2: keyword-level fuzzy match    (catches reworded/unseen codes)
                        └── Pass 3: Gemini triage (last resort)  (only genuinely ambiguous cases)
        │
        ▼
Pick recovery action  (matched to predicted cause — or escalate, never auto-retry risk blocks)
        │
        ▼
Simulate outcome  (success probability depends on the TRUE cause, not the agent's guess)
        │
        ▼
Draft customer message  (Gemini, ONE batched call for all actionable cases)
        │
        ▼
Report: recovery rate by category + honest exception list
```

### Why AI is used where it is (and not everywhere)

Most of this pipeline is deterministic, auditable logic — cheap and fast. AI is used deliberately in exactly two bounded places:

- **Classification (Gemini triage):** only for cases the rule-based + fuzzy classifier genuinely can't resolve. In testing, this was 1 out of 70 cases — the rest were resolved for free.
- **Generation (customer messages):** drafts the actual recovery nudge text for every actionable case, but batched into a **single API call** rather than one call per transaction.

This reflects a deliberate design position: generating an answer is the easy part — knowing *when* AI is worth using, and using it efficiently, is the harder and more valuable part.

## Test results (on the bundled synthetic dataset)

- **Classifier accuracy:** 95.7% (67/70) — 57 resolved by exact rule match, 12 by fuzzy fallback, 1 by Gemini triage
- **Overall recovery rate:** ~85% of at-risk amount recovered
- **Risk-block cases correctly routed to manual review** instead of auto-retried — recovery rate for this category is deliberately lower (~30%), reflecting safe behavior rather than a weakness
- **Honest exception list** included in every run — unresolved cases are reported with the reason, never hidden

## Project structure

```
payment-recovery-agent/
├── app/
│   └── app.py              # Streamlit dashboard
├── src/
│   ├── generate_dataset.py # Synthetic failed-payment dataset + ground truth
│   ├── classify.py         # 2-pass classifier (rule-based + fuzzy)
│   ├── llm_triage.py       # Gemini triage for unresolved cases
│   ├── recover.py          # Recovery action selection + simulated execution
│   └── message_gen.py      # Batched Gemini call for customer messages
├── data/                   # Generated CSVs (dataset, ground truth, results)
├── requirements.txt
└── README.md
```

## Running it locally

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

# 2. Add your Gemini API key
# Create a .env file in the project root:
#   GEMINI_API_KEY=your_key_here
# (Free tier available at aistudio.google.com — no card required)

# 3. Generate the dataset (or use the bundled one in data/)
python src/generate_dataset.py

# 4. Run the pipeline manually (optional — the dashboard does this for you)
python src/classify.py
python src/llm_triage.py
python src/recover.py

# 5. Launch the dashboard
streamlit run app/app.py
```

Click **Run Recovery Pipeline** in the sidebar. Upload your own `failed_payments.csv` (same schema as the bundled sample) to test against different data, or use the default.

## Honest limitations

- Recovery outcomes are **simulated**, not connected to a live payment gateway — a natural next step would be integrating with Razorpay's test-mode APIs for actual retry execution.
- The fuzzy-match confidence threshold (60%) is a tunable trade-off: tightening it sends more borderline cases to Gemini (more accurate, more cost); loosening it rescues more paraphrases but risks more wrong guesses. In testing, 2 of 3 deliberately ambiguous cases were misclassified with moderate confidence rather than correctly falling through to LLM triage — a known, documented trade-off, not a hidden gap.
- The LLM triage step currently calls Gemini once per unresolved case rather than batching (unlike the message-drafting step) — a straightforward follow-up optimization.

## Tech stack

Python, pandas, rapidfuzz, Streamlit, Google Gemini API (`google-genai`), python-dotenv
