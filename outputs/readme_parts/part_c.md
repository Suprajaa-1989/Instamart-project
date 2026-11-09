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
