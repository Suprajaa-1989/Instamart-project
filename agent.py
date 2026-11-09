
import datetime as dt
import json
import os
import sqlite3
import time

import mock_llm
import rag
from util import write

OUT, STATE, DB = "outputs", "state", "data/instamart.db"
LEDGER = f"{OUT}/refund_ledger.jsonl"
os.makedirs(OUT, exist_ok=True)
os.makedirs(STATE, exist_ok=True)

# =================================================================== task 5: tools
class SimulatedCrash(Exception):
    pass


def get_order(order_id):
    con = sqlite3.connect(DB)
    r = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if r is None:
        return {"error": "order not found"}
    return {"order_id": r[0], "status": r[1], "order_amount": r[2],
            "delivery_minutes": None if r[3] is None or r[3] != r[3] else r[3]}


def search_policy(query):
    return {"results": rag.retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")


def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def _spec(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}}


TOOLS_SPEC = [
    _spec("get_order", "Read an order's status, amount and delivery time from the database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the Instamart policy documents.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Apply the refund rules to an order amount and delivery time.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue a refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

AGENT_SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools get_order, search_policy, calculate_refund, "
                "issue_refund and escalate_to_human, one step at a time. End with a final answer that starts with "
                "REFUND, ESCALATE, DENIED or NOT_FOUND followed by a colon.")


# =================================================================== tasks 6-8: ReAct loop, limits, checkpoints
def _checkpoint(st):
    write(f"{STATE}/{st['run_id']}.json", json.dumps(st, indent=2, default=str))


def _finish(st, status, final=None):
    if final is None:
        final = (f"Human handoff: run stopped with {status}. Steps completed: "
                 f"{', '.join(st['tools_executed']) or 'none'}.")
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final,
            "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}


def _loop(st, crash_after_step=None):
    start = time.monotonic()
    lim = st["limits"]
    while st["llm_calls"] < lim["max_iterations"]:
        if time.monotonic() - start >= lim["time_limit_seconds"]:          # (c) wall-clock guard, before the model call
            return _finish(st, "FALLBACK_TIMEOUT")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > lim["token_limit"]:                         # (b) token guard, do not run the tool
            return _finish(st, "FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        if not calls:                                                       # final answer
            final = msg.content or ""
            status = final.split(":")[0].strip().upper() if ":" in final else "DENIED"
            return _finish(st, status, final)
        tc = calls[0]
        args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        try:
            obs = TOOL_MAP[tc.name](**args)
        except Exception as e:                                              # unknown tool / bad args -> observation
            obs = {"error": f"{type(e).__name__}: {e}"}
        # 'name' is added because the supplied mock reads the tool name from the tool message
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["step"] += 1
        st["tools_executed"].append(tc.name)
        st["steps"].append({"step": st["step"], "tool": tc.name, "arguments": args, "observation": obs})
        _checkpoint(st)                                                     # task 8: after every observation
        if crash_after_step is not None and st["step"] == crash_after_step:
            raise SimulatedCrash(f"simulated crash after tool execution {st['step']}")
    return _finish(st, "FALLBACK_MAX_ITERATIONS")                           # (a)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_step=None):
    st = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(st, crash_after_step)


def resume_agent(run_id):
    st = json.load(open(f"{STATE}/{run_id}.json", encoding="utf-8"))
    return _loop(st)


def refund_of(result):
    amts = [s["observation"].get("refund_inr") for s in result["steps"] if s["tool"] == "calculate_refund"]
    return amts[-1] if amts else 0.0


