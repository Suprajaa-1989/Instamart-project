import csv
import datetime as dt
import glob
import json
import os
import shutil
import sqlite3
import time

import pandas as pd

import mock_llm
import rag
from util import md_table, write

OUT, REP, STATE, DB = "outputs", "outputs/readme_parts", "state", "data/instamart.db"
RUNS, LEDGER = f"{OUT}/agent_runs.jsonl", f"{OUT}/refund_ledger.jsonl"
os.makedirs(REP, exist_ok=True)
shutil.rmtree(STATE, ignore_errors=True)
os.makedirs(STATE)
for f in (RUNS, LEDGER):
    open(f, "w").close()

# =================================================================== tasks 1-4: chunking, retrieval, grounding
chunks10 = rag.chunk_documents(40, 10)
per_doc = pd.Series([c["source"] for c in chunks10]).value_counts().sort_index()
ret = rag.Retriever(40, 10)
top_late = [h["source"] for h in ret.retrieve("refund policy late delivery")]
tfidf_note = ("TF is how often a term appears in a chunk and IDF down-weights terms that appear in many chunks, so rare, distinctive words drive the score. "
              "Its limitation here is that it is purely lexical: 'money back' and 'refund' mean the same but share no tokens, so a differently worded question can miss the right chunk.")

ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
q_col, doc_col, fact_col = ev.columns[0], ev.columns[1], ev.columns[2]


def run_eval(retriever):
    rows = []
    for _, r in ev.iterrows():
        text, sources = retriever.answer(r[q_col])
        if r[doc_col] == "NONE":
            rows.append({"question": r[q_col], "answerable": False, "retrieved": None, "grounded": None,
                         "fallback": text.startswith("I could not find")})
        else:
            rows.append({"question": r[q_col], "answerable": True, "retrieved": r[doc_col] in sources,
                         "grounded": r[fact_col] in text, "fallback": None})
    d = pd.DataFrame(rows)
    a, u = d[d.answerable], d[~d.answerable]
    return d, {"answerable": len(a), "expected_doc_in_top3": int(a.retrieved.sum()),
               "answers_with_expected_fact": int(a.grounded.sum()),
               "unanswerable": len(u), "correct_fallback": int(u.fallback.sum())}


ev10, s10 = run_eval(ret)
ret0 = rag.Retriever(40, 0)
ev0, s0 = run_eval(ret0)
broken = ev0[ev0.answerable & ~(ev0.retrieved & ev0.grounded)]
broken_note = "none"
if len(broken):
    qn = broken.iloc[0]["question"]
    row = ev[ev[q_col] == qn].iloc[0]
    in_one_chunk = any(row[fact_col] in c["text"] for c in ret0.chunks if c["source"] == row[doc_col])
    broken_note = (f"\"{qn}\" breaks with zero overlap: "
                   + ("the expected fact is split across a 40-word chunk boundary, so no single chunk holds the whole sentence."
                      if not in_one_chunk else "the chunk holding the fact no longer ranks in the top three."))

from agent import (SimulatedCrash, TOOLS_SPEC, TOOL_MAP, resume_agent, refund_of, run_agent)  # noqa: E402

# ---- run the seven scenarios
sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
sc_rows = []
for _, r in sc.iterrows():
    sid, msg, exp_out, exp_amt = r.iloc[0], r.iloc[2], r.iloc[3], float(r.iloc[4] or 0)
    res = run_agent(msg, sid)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")
    amt = refund_of(res)
    sc_rows.append({"scenario": sid, "status": res["status"], "expected": exp_out, "refund_inr": round(amt, 2),
                    "expected_refund_inr": exp_amt, "amount_ok": abs(amt - exp_amt) <= 0.01,
                    "model_calls": res["llm_calls"], "tokens": res["tokens_used"]})
sc_df = pd.DataFrame(sc_rows)
s1_msg = sc.iloc[0, 2]

# ---- stop-safely demos
d_tok = run_agent(s1_msg, "S1", token_limit=200, run_id="S1_token_limit_demo")
d_time = run_agent(s1_msg, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")

# ---- crash and resume
try:
    run_agent(s1_msg, "S1", run_id="S1_crash_demo", crash_after_step=2)
    raise RuntimeError("crash demo did not crash")
except SimulatedCrash:
    pass
resumed = resume_agent("S1_crash_demo")
executed = json.load(open(f"{STATE}/S1_crash_demo.json"))["tools_executed"]
assert executed == ["get_order", "search_policy", "calculate_refund", "issue_refund"], executed
assert resumed["status"] == "REFUND"
if abs(refund_of(resumed) - 128.25) > 0.01:
    print(f"WARNING: S1 refund is {refund_of(resumed)}, reference is 128.25")

# ---- ReAct vs plan-and-execute (<=250 words), numbers from the runs above
by = sc_df.set_index("scenario")
c = lambda s: int(by.loc[s, "model_calls"]) if s in by.index else 0
t = lambda s: int(by.loc[s, "tokens"]) if s in by.index else 0
react_text = (
    f"Measured: the ReAct agent needed {c('S1')} model calls and {t('S1')} tokens for S1 (happy path), {c('S3')} calls for the escalated S3, and only "
    f"{c('S6')} calls ({t('S6')} tokens) for the unknown order in S6, because each observation is fed back and the model stops as soon as it sees NOT_FOUND. "
    f"A plan-and-execute design asks the model for the whole step list once, runs the tools with no further model calls, then writes one final answer: "
    f"roughly 2 calls on the happy path (an estimate, not measured), fewer tokens than ReAct's {c('S1')} calls. "
    f"But when an observation such as NOT_FOUND or a cancelled order invalidates the plan, it must replan, costing about 3 calls even for a rejected request, and a stale plan could issue a refund before the check that should have blocked it. "
    f"For this refund workflow I would choose ReAct: the path depends on what each tool returns (status, delivery time, the INR 1000 threshold), "
    f"money moves at the end, and the extra calls are cheap compared with an incorrect refund. "
    f"Plan-and-execute becomes attractive only if volume grows and the policy checks are moved into deterministic code.")
assert len(react_text.split()) <= 250, len(react_text.split())

# =================================================================== README section
tools_all = executed
section = f"""## Part D - Policy RAG Assistant and Refund Agent

**Chunking (40 words, overlap 10).** {len(chunks10)} chunks in total: {', '.join(f"{k.replace('_policy.txt','')} {v}" for k, v in per_doc.items())}.

**Retrieval.** `TfidfVectorizer` fitted on the chunk texts, cosine similarity, `retrieve(query, k=3)` returns source, score and text.
For `refund policy late delivery` the top three chunks come from: {', '.join(top_late)}. {tfidf_note}

**Grounded answers and evaluation (14 questions).**

{md_table(pd.DataFrame([{"chunking": "overlap 10", **s10}, {"chunking": "overlap 0", **s0}]))}

Zero-overlap failure: {broken_note}

**Tools.** `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human` (specs in `TOOLS_SPEC`, dispatch in `TOOL_MAP`). Rules: only completed orders; delivery > 45 min -> 100% refund; <= 45 -> 0; refund above INR 1000 -> escalated to a human.

**ReAct loop and the seven scenarios** (`outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`):

{md_table(sc_df)}

**Stop-safely demonstrations.** `token_limit=200` on S1 ended with status **{d_tok['status']}** ({d_tok['llm_calls']} model calls, {d_tok['tokens_used']} tokens);
`time_limit_seconds=0` on S1 ended with status **{d_time['status']}** ({d_time['llm_calls']} model calls). S7 ends with `FALLBACK_MAX_ITERATIONS` under the default `token_limit=6000`.
Fallbacks return the steps completed so far in `final_answer` as a human-handoff summary.

**Crash and resume (S1).** A simulated crash after the second tool execution (checkpoint already saved in `state/S1_crash_demo.json`), then `resume_agent("S1_crash_demo")`.
Tools executed across both parts, in order: {' -> '.join(tools_all)} (four executions, none repeated). Final status **{resumed['status']}**, refund INR {refund_of(resumed):.2f}.

**ReAct versus plan-and-execute.** {react_text}
"""
write(f"{REP}/part_d.md", section)
print(f"Part D done: scenarios ok={bool(sc_df.amount_ok.all())}, crash/resume tools={executed}")
