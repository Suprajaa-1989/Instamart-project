# Instamart Customer Intelligence & Support Copilot

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
source .venv/bin/activate          # Windows: .venv\Scripts\activate
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

## Part A - Data Foundation

**Raw data audit (task 2).** Shapes: customers 1200x5, products 40x4, orders 25576x9, tickets 164x5.

| audit figure | value |
|---|---|
| orders raw rows | 25576 |
| exact duplicate rows | 227 |
| rows with status 'pending' | 1227 |
| distinct order_id | 24122 |
| order_amount not convertible to a number | 224 |
| negative delivery_minutes | 0 |
| blank delivery_minutes | 8078 |
| customers: blank age | 65 |
| customers: age outside 10-100 | 385 |
| customers: distinct raw city spellings | 18 |
| customers: distinct non-blank raw gender spellings | 4 |
| customers: blank gender | 222 |

Per-column dtypes and missing values (count and %):

**customers**

| column | dtype | missing | missing_pct |
|---|---|---|---|
| customer_id | str | 0 | 0.0 |
| city | str | 0 | 0.0 |
| signup_date | str | 0 | 0.0 |
| age | float64 | 65 | 5.42 |
| gender | str | 222 | 18.5 |

**products**

| column | dtype | missing | missing_pct |
|---|---|---|---|
| product_id | str | 0 | 0.0 |
| product_name | str | 0 | 0.0 |
| category | str | 0 | 0.0 |
| price_inr | int64 | 0 | 0.0 |

**orders**

| column | dtype | missing | missing_pct |
|---|---|---|---|
| order_id | str | 0 | 0.0 |
| customer_id | str | 0 | 0.0 |
| product_id | str | 0 | 0.0 |
| quantity | int64 | 0 | 0.0 |
| order_ts | str | 0 | 0.0 |
| status | str | 0 | 0.0 |
| discount_pct | int64 | 0 | 0.0 |
| order_amount | str | 110 | 0.43 |
| delivery_minutes | float64 | 8078 | 31.58 |

**tickets** (read with `keep_default_na=False`; blank means empty string)

| column | dtype | missing | missing_pct |
|---|---|---|---|
| ticket_id | str | 0 | 0.0 |
| customer_id | str | 0 | 0.0 |
| order_id | str | 164 | 100.0 |
| text | str | 4 | 2.44 |
| true_category | str | 0 | 0.0 |

**Cleaning rules and how many rows/values each changed (tasks 3-4).**

| rule | rows / values changed |
|---|---|
| customers: city spellings -> clean cities | 18 -> 6 |
| customers: gender values mapped to male/female/unknown | 4 spellings + 222 blanks -> 3 values |
| customers: blank ages imputed with city median | 65 |
| customers: ages outside 10-100 set to missing, then imputed | 385 |
| orders: exact duplicate rows removed | 227 |
| orders: superseded (earlier 'pending') records removed, latest order_ts kept | 1227 |
| orders: order_amount unparseable -> recomputed from price x qty x (1-disc) | 216 |
| orders: negative delivery_minutes set to missing | 0 |
| orders: completed orders' missing delivery_minutes -> category median (completed) | 4307 |
| orders: cancelled orders' delivery_minutes left missing | 2477 |

| after-cleaning check | this run | reference |
|---|---|---|
| orders rows (one per order_id) | 24122 | 12,309 |
| completed / cancelled | 21,645 / 2,477 | 11,110 / 1,199 |
| pending rows | 0 | 0 |
| missing order_amount | 0 | 0 |
| missing delivery_minutes (completed / cancelled) | 0 / 2477 | 0 / 1,199 |
| sum of order_amount, completed (INR) | 21,403,983.10 | 2,998,668.80 |
| customers: ages missing / min-max age | 0 / 25-50 | 0 / 18-60 |
| customers: gender counts | female 493, male 485, unknown 222 | female 610, male 560, unknown 30 |
| customers: city counts | Pune 218, Chennai 207, Hyderabad 202, Delhi 201, Bengaluru 191, Mumbai 181 | 6 cities |

**Storage and validation (task 5).** Clean tables are in `data/clean/*.csv` and in SQLite `data/instamart.db`
(tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text).
`python validate_part_a.py` asserts the reference checkpoints and prints `ALL CHECKPOINTS PASSED`.

**SQL analysis (task 6)** - queries in `sql/queries.sql`, executed with `pd.read_sql_query`.

| query | rows returned | reference / key result |
|---|---|---|
| Q1 WHERE then HAVING | 885 | 30 customers |
| Q2 three-table join | 30 | 30 rows; largest = Pune / Grocery INR 928,992.00 (ref. Bengaluru / Electronics 303,909.45) |
| Q3 scalar subquery | 21 | 11 products |
| Q4 LEFT JOIN | 26 | 25 customers |
| Q5 CTE | 592 | 550 customers; average INR 18,231.67 (ref. 2,552.06) |
| Q6 window functions | 1174 | 2 rows where RANK() differs from ROW_NUMBER() (ref. 8) |
| Q7 running total | 183 | 183 daily rows; last cumulative INR 21,403,983.10 (ref. 2,998,668.80) |

Top customer per city (Q6): 

| city | customer_id | total_spend |
|---|---|---|
| Bengaluru | C1184 | 48484.6 |
| Chennai | C0659 | 44902.2 |
| Delhi | C0468 | 41745.0 |
| Hyderabad | C0967 | 46512.0 |
| Mumbai | C0387 | 43364.6 |
| Pune | C1135 | 45430.8 |

Q6 note: RANK() and ROW_NUMBER() differ in exactly 2 rows because customers who tie on total spend share the same RANK() (and the next rank is skipped) while ROW_NUMBER() always assigns a unique number.

**pandas cross-check (task 7a).** `pd.pivot_table(..., margins=True)` of completed revenue by city x category gives a grand total of INR 21,403,983.10, equal to the SQL Q2 total.

| city | Dairy | Electronics | Grocery | Personal Care | Snacks | Total |
|---|---|---|---|---|---|---|
| Bengaluru | 736861.7 | 701236.8 | 667399.4 | 779261.2 | 538773.4 | 3423532.5 |
| Chennai | 637076.4 | 702788.4 | 764723.9 | 788845.9 | 632044.7 | 3525479.3 |
| Delhi | 650726.4 | 712134.7 | 737359.0 | 830296.1 | 604846.2 | 3535362.4 |
| Hyderabad | 700219.2 | 702721.9 | 745315.4 | 861818.0 | 612780.6 | 3622855.1 |
| Mumbai | 650239.1 | 633581.3 | 689621.1 | 740653.2 | 587878.0 | 3301972.7 |
| Pune | 688261.1 | 786644.5 | 928992.0 | 920875.1 | 670008.4 | 3994781.1 |
| Total | 4063383.9 | 4239107.6 | 4533410.8 | 4921749.5 | 3646331.3 | 21403983.1 |

Per-city aggregation (`groupby` + named aggregation):

| city | order_sum | order_mean | order_count |
|---|---|---|---|
| Bengaluru | 3423532.5 | 1006.92 | 3400 |
| Chennai | 3525479.3 | 994.21 | 3546 |
| Delhi | 3535362.4 | 983.14 | 3596 |
| Hyderabad | 3622855.1 | 974.67 | 3717 |
| Mumbai | 3301972.7 | 979.81 | 3370 |
| Pune | 3994781.1 | 994.72 | 4016 |

**EDA (task 7b).** Stats of completed `order_amount`: mean = 988.87, median = 898.00, skewness = 0.66, Q1 = 359.20, Q3 = 1,465.60, lower fence = -1,300.40, upper fence = 3,125.20, outliers above = 0, outliers below = 0.

Correlation matrix (completed orders):

| index | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.0 | -0.109 | 0.439 | -0.01 |
| discount_pct | -0.109 | 1.0 | 0.006 | -0.004 |
| quantity | 0.439 | 0.006 | 1.0 | 0.006 |
| delivery_minutes | -0.01 | -0.004 | 0.006 | 1.0 |

**What is this data?** 21,645 completed Instamart orders from 01 Apr 2025 to 30 Sep 2025, with amount (INR), quantity, discount and delivery time. **Distribution:** order_amount is right-skewed (mean 988.87 vs median 898.00, skewness 0.66); half of orders lie between 359.20 and 1465.60 at the quartiles. **Outliers:** 0 orders exceed the upper 1.5xIQR fence of 3125.20 and 0 fall below the lower fence; they look like genuine high-priced baskets (likely electronics), not errors. **Relationships:** the largest absolute correlation is order_amount with quantity (r = +0.439); r of this size is weak and does not establish causation.

All amounts in this project are in INR.

## Part B - Customer Segments and Lapse Prediction

**RFM table (snapshot 2025-10-01).** 1,174 customers; recency median 7, min 0, frequency mean 18.44, max 39, monetary median 18,376.80, max 48,484.60.

**K-means sweep (standardised RFM, `n_init=10`, `random_state=42`).**

| K | inertia | silhouette |
|---|---|---|
| 2.0 | 1574.8 | 0.501 |
| 3.0 | 899.1 | 0.499 |
| 4.0 | 602.7 | 0.446 |
| 5.0 | 475.9 | 0.415 |
| 6.0 | 388.7 | 0.377 |
| 7.0 | 329.5 | 0.38 |
| 8.0 | 288.8 | 0.342 |

I chose **K=3**: it has the highest silhouette among K in {3,4,5} and gives segments the business can act on. Inertia always falls as K grows
(more centroids can only reduce within-cluster distance, reaching 0 at K=n), so K=8 beating K=2 on inertia says nothing by itself; the elbow and silhouette decide.

**Segment profile.**

| segment_id | size | mean_recency | mean_frequency | mean_monetary | segment_name | recommended_action | is_premium |
|---|---|---|---|---|---|---|---|
| 0 | 551 | 5.6 | 27.93 | 27922.06 | Premium High-Spenders | Early access, bundles and a dedicated premium desk | True |
| 1 | 114 | 76.99 | 3.82 | 3687.76 | Lapsing / Dormant | Win-back coupon before they churn | False |
| 2 | 509 | 13.95 | 11.44 | 10999.06 | Active Regulars | Loyalty rewards and cross-sell of adjacent categories | False |

The segment with the highest mean monetary value is flagged **Premium**. Cluster numbers are arbitrary labels (no ordering).
PCA(2) on the scaled RFM matrix: explained variance ratio 0.791 + 0.194 = 0.985.
`outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

**Lapse problem (cutoff 2025-09-01).** Cohort: 1,169 customers with a completed order before the cutoff; 187 lapsed (16.00%).
Split: 935 train / 234 test (`stratify=y`, `random_state=42`). Always predicting "not lapsed": accuracy 0.842, lapsed-class F1 0.000.
**No order dated on or after 2025-09-01 entered the features, and the segment labels from Part B task 3 were not used as features.**

**Leakage-free preprocessing.** One-hot (`drop="first"`) and `StandardScaler` live inside a scikit-learn `Pipeline`; no encoder, scaler or resampler saw test rows.
Scaler mean of `frequency` (train only) = 15.446 vs full-cohort mean 15.538: they differ because the scaler only sees the 80% training sample. No feature column has a missing value (asserted in code).

**Models (5-fold CV, scoring F1).**

Logistic regression grid (best `C` = 10, CV F1 0.454):

| C | mean_test_score |
|---|---|
| 100.0 | 0.4542 |
| 10.0 | 0.4542 |
| 1.0 | 0.4404 |
| 0.1 | 0.3713 |
| 0.01 | 0.1337 |

Random forest grid (best {'clf__max_depth': 8, 'clf__min_samples_leaf': 10}, CV F1 0.488):

| max_depth | min_samples_leaf | mean_test_score |
|---|---|---|
| 8.0 | 10.0 | 0.4877 |
| 8.0 | 5.0 | 0.456 |
| 5.0 | 10.0 | 0.4204 |
| 5.0 | 5.0 | 0.4203 |
| 8.0 | 20.0 | 0.3985 |
| 5.0 | 20.0 | 0.3842 |
| 3.0 | 5.0 | 0.3008 |
| 3.0 | 10.0 | 0.2798 |
| 3.0 | 20.0 | 0.2689 |

Test set, lapsed class:

| model | confusion matrix | precision | recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[TN 189, FP 8], [FN 22, TP 15]] | 0.652 | 0.405 | 0.5 |
| Random forest | [[TN 190, FP 7], [FN 22, TP 15]] | 0.682 | 0.405 | 0.508 |

I select the random forest (best cross-validated F1 0.488, test F1 0.508, recall 0.405, precision 0.682). Missing a lapsing customer (a false negative) costs a lost future revenue stream that a cheap voucher could have saved, while wrongly targeting a loyal customer (a false positive) only wastes one small discount. So recall on the lapsed class matters most, but F1 keeps precision from collapsing and the retention budget under control. The model chosen gives the best balance of the two on unseen customers.

**Class-balance experiment (training set only).** Added 635 random duplicates of lapsed training rows (`random_state=42`) so both classes have 785 rows; refit the best logistic regression and tested on the original test set.

| training data | precision | recall | F1 |
|---|---|---|---|
| unbalanced | 0.652 | 0.405 | 0.5 |
| oversampled | 0.459 | 0.919 | 0.613 |

**Recall** moved most. For retention this means oversampling lets us catch more lapsing customers at roughly the same precision, which suits a cheap-to-send campaign.

**Interpretation.** Top-5 random-forest features (`feature_importances_`):

| index | importance |
|---|---|
| monetary | 0.3025 |
| frequency | 0.2617 |
| recency | 0.1039 |
| cancel_rate | 0.0876 |
| avg_discount_pct | 0.0871 |

Top-3 logistic-regression coefficients: `frequency` (-1.364): lower lapse risk when this value is higher / present, so this behaviour signals loyalty; `monetary` (-0.886): lower lapse risk when this value is higher / present, so this behaviour signals loyalty; `gender_unknown` (+0.569): higher lapse risk when this value is higher / present, so watch customers with a high value here.

Model saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading with `joblib.load` reproduces the test predictions exactly (asserted).
**Retention list.** Test-set customers in the Premium segment: 113. Top 10 by predicted lapse probability are in `outputs/retention_list.csv`; 0 of them truly lapsed.
The Premium flag comes from whole-window RFM (it includes the September window that defines the label), so it is a post-hoc whole-window label and the retention list is illustrative, not a leakage-free evaluation.

## Part C - Support Ticket Triage with Guardrails

**Input guardrails.** 4 empty tickets -> `skipped_empty`; 4 tickets with `est_tokens(text) > 400` -> `flagged_long`. Neither is ever sent to the model.

**PII redaction (applied in order email -> phone -> order id).**

| pattern | substitutions |
|---|---|
| email | 37 |
| phone | 31 |
| order_id | 112 |
| total | 180 |

Re-applying the three patterns to the redacted texts finds 0 matches, and an `assert` before every model call checks the prompt; the user message is exactly the redacted ticket text.

**Prompts.** Zero-shot: system prompt with the word `CLASSIFY`, the four categories and the JSON-only format. Few-shot: the same plus a line `EXAMPLES:` followed by four labelled examples (billing, delivery, technical, fraud).

**Retries.** `TransientLLMError` is retried up to 3 times with wait = 0.05 x 2^n x uniform(0.9, 1.15); each retry is logged to `outputs/retry_log.txt`. The mock is reset before each run.
Few-shot run: 19 transient errors, all recovered (no ticket ended in a permanent error); zero-shot run: 19.

**Validation.** JSON with an allowed category -> `classified_json`; otherwise exactly one allowed word found by `re.findall(r"[a-z]+", text.lower())` -> `classified_keyword`; anything else (including valid JSON with an out-of-list category) -> `flagged_invalid_output` and human review, never guessed.

**Zero-shot vs few-shot (no budget).**

| prompt | classified | accuracy | total_tokens | tokens_per_classified_ticket | classified_json | skipped_empty | flagged_long |
|---|---|---|---|---|---|---|---|
| zero-shot | 156 | 26.9 | 10380 | 66.5 | 156 | 4 | 4 |
| few-shot | 156 | 26.9 | 27852 | 178.5 | 156 | 4 | 4 |

Confusion matrix, zero-shot:

| true | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 7 | 9 | 16 | 7 |
| delivery | 10 | 8 | 9 | 12 |
| fraud | 14 | 5 | 11 | 9 |
| technical | 6 | 9 | 8 | 16 |

Few-shot costs 178.5 vs 66.5 tokens per classified ticket (+112.0, i.e. +168%) because the four examples are re-sent on every call. In return accuracy rises from 26.9% to 26.9% and the mock stops confusing fraud with billing and technical tickets, which is the costliest mistake since fraud must reach the security team. The extra tokens are worth paying for when misrouted fraud tickets are expensive; for low-stakes categories zero-shot would do.

**Token budget.** With `TOKEN_BUDGET = 6000` the few-shot pipeline used 6073 tokens and set 122 tickets to `budget_exhausted` (no model call; routed to human review). Saved in `outputs/triage_budget_demo.csv`.

**Routing** (`outputs/triage_results.csv`, 164 rows, few-shot, `TOKEN_BUDGET = 100000`): human statuses -> `human_review` NORMAL; fraud -> `security_team` HIGH; Premium customers (Part B segment file) -> `premium_desk` HIGH; otherwise `support_queue` NORMAL. Customers missing from the segment file are treated as not premium.

| route | priority | tickets |
|---|---|---|
| human_review | NORMAL | 8 |
| premium_desk | HIGH | 47 |
| security_team | HIGH | 44 |
| support_queue | NORMAL | 65 |

**Cost and trust.** The few-shot prompt costs about 178.5 tokens per classified ticket. Rule-validated outputs (`classified_json`, `classified_keyword`) are routed automatically;
`skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` tickets go to humans.

## Part D - Policy RAG Assistant and Refund Agent

**Chunking (40 words, overlap 10).** 6 chunks in total: delivery 3, refund 3.

**Retrieval.** `TfidfVectorizer` fitted on the chunk texts, cosine similarity, `retrieve(query, k=3)` returns source, score and text.
For `refund policy late delivery` the top three chunks come from: delivery_policy.txt, refund_policy.txt, refund_policy.txt. TF is how often a term appears in a chunk and IDF down-weights terms that appear in many chunks, so rare, distinctive words drive the score. Its limitation here is that it is purely lexical: 'money back' and 'refund' mean the same but share no tokens, so a differently worded question can miss the right chunk.

**Grounded answers and evaluation (14 questions).**

| chunking | answerable | expected_doc_in_top3 | answers_with_expected_fact | unanswerable | correct_fallback |
|---|---|---|---|---|---|
| overlap 10 | 1 | 1 | 1 | 1 | 1 |
| overlap 0 | 1 | 1 | 1 | 1 | 1 |

Zero-overlap failure: none

**Tools.** `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human` (specs in `TOOLS_SPEC`, dispatch in `TOOL_MAP`). Rules: only completed orders; delivery > 45 min -> 100% refund; <= 45 -> 0; refund above INR 1000 -> escalated to a human.

**ReAct loop and the seven scenarios** (`outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`):

| scenario | status | expected | refund_inr | expected_refund_inr | amount_ok | model_calls | tokens |
|---|---|---|---|---|---|---|---|
| S1 | REFUND | REFUND | 82.0 | 82.0 | True | 5 | 1067 |
| S2 | REFUND | REFUND | 82.0 | 82.0 | True | 5 | 1057 |
| S3 | ESCALATE | ESCALATE | 1185.0 | 1185.0 | True | 5 | 1062 |
| S4 | DENIED | DENIED | 0.0 | 0.0 | True | 4 | 749 |
| S5 | DENIED | DENIED | 0.0 | 0.0 | True | 3 | 445 |
| S6 | NOT_FOUND | NOT_FOUND | 0.0 | 0.0 | True | 2 | 148 |
| S7 | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | 0.0 | 0.0 | True | 6 | 3168 |

**Stop-safely demonstrations.** `token_limit=200` on S1 ended with status **FALLBACK_TOKEN_LIMIT** (3 model calls, 456 tokens);
`time_limit_seconds=0` on S1 ended with status **FALLBACK_TIMEOUT** (0 model calls). S7 ends with `FALLBACK_MAX_ITERATIONS` under the default `token_limit=6000`.
Fallbacks return the steps completed so far in `final_answer` as a human-handoff summary.

**Crash and resume (S1).** A simulated crash after the second tool execution (checkpoint already saved in `state/S1_crash_demo.json`), then `resume_agent("S1_crash_demo")`.
Tools executed across both parts, in order: get_order -> search_policy -> calculate_refund -> issue_refund (four executions, none repeated). Final status **REFUND**, refund INR 82.00.

**ReAct versus plan-and-execute.** Measured: the ReAct agent needed 5 model calls and 1067 tokens for S1 (happy path), 5 calls for the escalated S3, and only 2 calls (148 tokens) for the unknown order in S6, because each observation is fed back and the model stops as soon as it sees NOT_FOUND. A plan-and-execute design asks the model for the whole step list once, runs the tools with no further model calls, then writes one final answer: roughly 2 calls on the happy path (an estimate, not measured), fewer tokens than ReAct's 5 calls. But when an observation such as NOT_FOUND or a cancelled order invalidates the plan, it must replan, costing about 3 calls even for a rejected request, and a stale plan could issue a refund before the check that should have blocked it. For this refund workflow I would choose ReAct: the path depends on what each tool returns (status, delivery time, the INR 1000 threshold), money moves at the end, and the extra calls are cheap compared with an incorrect refund. Plan-and-execute becomes attractive only if volume grows and the policy checks are moved into deterministic code.

## Limitations and Next Steps

- The Premium flag comes from whole-window RFM, which includes the September window that defines the lapse label: it is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.
- The mock LLM is keyword-based, so the zero-shot/few-shot gap and agent behaviour are properties of the mock, not of a real model.
- Cleaning rules (city median age, category median delivery time) are simple imputations; a production pipeline would track imputation flags as features.
- Next steps: time-based cross-validation for the lapse model, calibrated probabilities, a real LLM behind the same `chat()` interface, and idempotency keys on `issue_refund`.
