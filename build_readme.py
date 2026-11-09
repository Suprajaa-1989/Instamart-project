"""Assemble README.md from the section files each Part wrote into outputs/readme_parts/."""
import os

from util import write

HEAD = """# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected Parts, all on synthetic data (INR) and an offline mock LLM (no API key, no network):

| Part | What it does | Consumes | Produces |
|---|---|---|---|
| A | Audit, clean, store (SQLite), SQL + pandas analysis | `data/raw/*.csv` | `data/clean/*`, `data/instamart.db`, `sql/queries.sql` |
| B | RFM segments (K-means, PCA) and lapse model | clean orders/customers | `outputs/customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv` |
| C | Ticket triage with guardrails and routing | raw tickets + **B's segment file** | `outputs/triage_results.csv`, `triage_budget_demo.csv` |
| D | Policy RAG + refund agent (ReAct, limits, resume) | policies + **A's database** | `outputs/agent_runs.jsonl`, `refund_ledger.jsonl`, `state/` |

`generate_data.py` and `mock_llm.py` are the supplied files, committed unmodified. `python run_all.py` runs everything in order.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python generate_data.py            # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python part_a.py
python validate_part_a.py          # prints ALL CHECKPOINTS PASSED
python part_b.py
python part_c.py
python part_d.py
python build_readme.py             # regenerates this README from the numbers above
```
(or simply `python run_all.py` after step 3). Random seeds are fixed (42) everywhere randomness appears.

"""
TAIL = """
## Limitations and Next Steps

- The Premium flag comes from whole-window RFM, which includes the September window that defines the lapse label: it is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.
- The mock LLM is keyword-based, so the zero-shot/few-shot gap and agent behaviour are properties of the mock, not of a real model.
- Cleaning rules (city median age, category median delivery time) are simple imputations; a production pipeline would track imputation flags as features.
- Next steps: time-based cross-validation for the lapse model, calibrated probabilities, a real LLM behind the same `chat()` interface, and idempotency keys on `issue_refund`.
"""
parts = []
for p in "abcd":
    path = f"outputs/readme_parts/part_{p}.md"
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run part_{p}.py first")
    parts.append(open(path, encoding="utf-8").read())
write("README.md", HEAD + "\n".join(parts) + TAIL)
print("README.md written")
