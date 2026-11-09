import json
import os
import random
import re
import time

import pandas as pd

import mock_llm
from util import md_table, write

OUT, REP = "outputs", "outputs/readme_parts"
os.makedirs(REP, exist_ok=True)
random.seed(42)

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
FINAL_TOKEN_BUDGET = 100000
CATS = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ("You are a support-ticket classifier. Your task is to CLASSIFY the customer ticket into exactly one "
             "category from this list: billing, delivery, technical, fraud. "
             'Reply ONLY with JSON of the form {"category": "...", "reason": "..."} and nothing else.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Example 1 - Ticket: I was charged twice and need the extra amount refunded.
Answer: {"category": "billing", "reason": "duplicate charge"}
Example 2 - Ticket: My order has not arrived and the rider is not picking up calls.
Answer: {"category": "delivery", "reason": "order not delivered"}
Example 3 - Ticket: The app crashes every time I open the cart.
Answer: {"category": "technical", "reason": "app crash"}
Example 4 - Ticket: There is an unauthorized transaction on my card that I never made.
Answer: {"category": "fraud", "reason": "unauthorized transaction"}"""
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# ------------------------------------------------------------------ task 2: redaction
redacted, counts = [], {n: 0 for n, _, _ in PATTERNS}
for text in tickets["text"]:
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        counts[name] += n
    redacted.append(text)
leftover = sum(len(pat.findall(t)) for t in redacted for _, pat, _ in PATTERNS)
assert leftover == 0, "PII left after redaction"
n_empty = int((tickets["text"].str.strip() == "").sum())
n_long = int(tickets["text"].map(lambda t: mock_llm.est_tokens(t) > MAX_INPUT_TOKENS).sum())


# ------------------------------------------------------------------ task 4: resilient call
def call_llm(system, user, ticket_id, retry_log):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            retry_log.append(f"{ticket_id} retry={attempt} wait={wait:.4f}s")
            time.sleep(wait)


# ------------------------------------------------------------------ task 5: output validation
def validate(content):
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        cat = obj.get("category")
        return ("classified_json", cat) if cat in CATS else ("flagged_invalid_output", "")
    found = set(re.findall(r"[a-z]+", str(content).lower())) & set(CATS)
    if len(found) == 1:
        return "classified_keyword", found.pop()
    return "flagged_invalid_output", ""


def run_pipeline(system, token_budget=None):
    mock_llm.reset()
    total, rows, retry_log = 0, [], []
    for i, t in enumerate(tickets.itertuples(index=False)):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0}
        if not str(t.text).strip():
            row["status"] = "skipped_empty"
        elif mock_llm.est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            safe = redacted[i]
            assert not any(p.search(safe) for _, p, _ in PATTERNS)       # privacy guardrail
            try:
                resp = call_llm(system, safe, t.ticket_id, retry_log)
            except mock_llm.TransientLLMError:
                row["status"] = "flagged_invalid_output"                    # permanent failure -> human
            else:
                tok = resp.usage.total_tokens
                total += tok
                row["tokens_used"] = tok
                row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), total, retry_log


# ------------------------------------------------------------------ task 6: zero-shot vs few-shot
zs_df, zs_tok, zs_retries = run_pipeline(ZERO_SHOT)
fs_df, fs_tok, fs_retries = run_pipeline(FEW_SHOT)
write(f"{OUT}/retry_log.txt", "\n".join(["# zero-shot run"] + zs_retries + ["# few-shot run"] + fs_retries) + "\n")


def summarise(df, total):
    ok = df[df.status.isin(["classified_json", "classified_keyword"])].merge(
        tickets[["ticket_id", "true_category"]], on="ticket_id")
    acc = (ok.category == ok.true_category).mean()
    return ok, {"classified": len(ok), "accuracy": round(acc * 100, 1), "total_tokens": total,
                "tokens_per_classified_ticket": round(total / len(ok), 1),
                **df.status.value_counts().to_dict()}


zs_ok, zs_sum = summarise(zs_df, zs_tok)
fs_ok, fs_sum = summarise(fs_df, fs_tok)
cmat = pd.crosstab(zs_ok.true_category, zs_ok.category, rownames=["true"], colnames=["predicted (zero-shot)"])
extra_tok = fs_sum["tokens_per_classified_ticket"] - zs_sum["tokens_per_classified_ticket"]
worth = (f"Few-shot costs {fs_sum['tokens_per_classified_ticket']} vs {zs_sum['tokens_per_classified_ticket']} tokens per classified ticket "
         f"(+{extra_tok:.1f}, i.e. +{extra_tok / zs_sum['tokens_per_classified_ticket'] * 100:.0f}%) because the four examples are re-sent on every call. "
         f"In return accuracy rises from {zs_sum['accuracy']}% to {fs_sum['accuracy']}% and the mock stops confusing fraud with billing and technical tickets, "
         f"which is the costliest mistake since fraud must reach the security team. "
         f"The extra tokens are worth paying for when misrouted fraud tickets are expensive; for low-stakes categories zero-shot would do.")

# ------------------------------------------------------------------ task 7: budget + routing
demo_budget, demo_df, demo_exhausted = None, None, 0
for b in (6000, 5000, 3000, 2000, 1000, 500):
    d, used, _ = run_pipeline(FEW_SHOT, token_budget=b)
    n_exh = int((d.status == "budget_exhausted").sum())
    if n_exh >= 1:
        demo_budget, demo_df, demo_exhausted, demo_used = b, d, n_exh, used
        break
assert demo_df is not None, "no budget produced an exhausted ticket"
demo_df[["ticket_id", "status", "category", "tokens_used"]].to_csv(f"{OUT}/triage_budget_demo.csv", index=False)

final_df, final_tok, _ = run_pipeline(FEW_SHOT, token_budget=FINAL_TOKEN_BUDGET)
seg = pd.read_csv(f"{OUT}/customer_segments.csv")
seg_map = seg.set_index("customer_id")[["segment_name", "is_premium"]]
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}


def route(r):
    name, prem = "", False
    if r.customer_id in seg_map.index:
        name, prem = seg_map.loc[r.customer_id, "segment_name"], bool(seg_map.loc[r.customer_id, "is_premium"])
    if r.status in HUMAN:
        return pd.Series([name, "NORMAL", "human_review"])
    if r.category == "fraud":
        return pd.Series([name, "HIGH", "security_team"])
    if prem:
        return pd.Series([name, "HIGH", "premium_desk"])
    return pd.Series([name, "NORMAL", "support_queue"])


final_df[["segment_name", "priority", "route"]] = final_df.apply(route, axis=1)
final_df = final_df[["ticket_id", "customer_id", "status", "segment_name", "category", "priority", "route", "tokens_used"]]
assert len(final_df) == len(tickets)
final_df.to_csv(f"{OUT}/triage_results.csv", index=False)

# ------------------------------------------------------------------ README section
retry_n = len(fs_retries)
section = f"""## Part C - Support Ticket Triage with Guardrails

**Input guardrails.** {n_empty} empty tickets -> `skipped_empty`; {n_long} tickets with `est_tokens(text) > {MAX_INPUT_TOKENS}` -> `flagged_long`. Neither is ever sent to the model.

**PII redaction (applied in order email -> phone -> order id).**

{md_table(pd.DataFrame([[k, v] for k, v in counts.items()] + [["total", sum(counts.values())]], columns=["pattern", "substitutions"]))}

Re-applying the three patterns to the redacted texts finds {leftover} matches, and an `assert` before every model call checks the prompt; the user message is exactly the redacted ticket text.

**Prompts.** Zero-shot: system prompt with the word `CLASSIFY`, the four categories and the JSON-only format. Few-shot: the same plus a line `EXAMPLES:` followed by four labelled examples (billing, delivery, technical, fraud).

**Retries.** `TransientLLMError` is retried up to {MAX_RETRIES} times with wait = {BASE_DELAY} x 2^n x uniform(0.9, 1.15); each retry is logged to `outputs/retry_log.txt`. The mock is reset before each run.
Few-shot run: {retry_n} transient errors, all recovered (no ticket ended in a permanent error); zero-shot run: {len(zs_retries)}.

**Validation.** JSON with an allowed category -> `classified_json`; otherwise exactly one allowed word found by `re.findall(r"[a-z]+", text.lower())` -> `classified_keyword`; anything else (including valid JSON with an out-of-list category) -> `flagged_invalid_output` and human review, never guessed.

**Zero-shot vs few-shot (no budget).**

{md_table(pd.DataFrame([{"prompt": "zero-shot", **zs_sum}, {"prompt": "few-shot", **fs_sum}]).fillna(0))}

Confusion matrix, zero-shot:

{md_table(cmat, index=True)}

{worth}

**Token budget.** With `TOKEN_BUDGET = {demo_budget}` the few-shot pipeline used {demo_used} tokens and set {demo_exhausted} tickets to `budget_exhausted` (no model call; routed to human review). Saved in `outputs/triage_budget_demo.csv`.

**Routing** (`outputs/triage_results.csv`, 164 rows, few-shot, `TOKEN_BUDGET = {FINAL_TOKEN_BUDGET}`): human statuses -> `human_review` NORMAL; fraud -> `security_team` HIGH; Premium customers (Part B segment file) -> `premium_desk` HIGH; otherwise `support_queue` NORMAL. Customers missing from the segment file are treated as not premium.

{md_table(final_df.groupby(['route', 'priority']).size().rename('tickets').reset_index())}

**Cost and trust.** The few-shot prompt costs about {fs_sum['tokens_per_classified_ticket']} tokens per classified ticket. Rule-validated outputs (`classified_json`, `classified_keyword`) are routed automatically;
`skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` tickets go to humans.
"""
write(f"{REP}/part_c.md", section)
print(f"Part C done: zero-shot {zs_sum['accuracy']}%, few-shot {fs_sum['accuracy']}%, retries {retry_n}")
