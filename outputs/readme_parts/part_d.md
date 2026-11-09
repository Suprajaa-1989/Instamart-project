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
