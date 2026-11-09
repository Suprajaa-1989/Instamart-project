"""STAND-IN mock LLM - only used when the official mock_llm.py is missing. Not for submission."""
import json, math, re
from types import SimpleNamespace as NS
class TransientLLMError(Exception): pass
_s={"calls":0,"ok":0}
def reset(): _s["calls"]=0;_s["ok"]=0
def est_tokens(t): return math.ceil(len((t or "").split())*1.3)
def _r(content=None,tc=None,messages=()):
    p=sum(est_tokens(m.get("content") or json.dumps(m.get("tool_calls"))) for m in messages)
    c=est_tokens(content or json.dumps(tc or ""))
    return NS(choices=[NS(message=NS(role="assistant",content=content,tool_calls=[NS(**x) for x in tc] if tc else None),finish_reason="stop")],usage=NS(prompt_tokens=p,completion_tokens=c,total_tokens=p+c))
KW={"fraud":{"unauthorized","fraud"},"technical":{"app","crash","login"},"billing":{"charged","refund"},"delivery":{"arrived","rider"}}
def chat(messages,tools=None):
    h=messages[0]["content"].upper()
    if "CLASSIFY" in h:
        _s["calls"]+=1
        if _s["calls"]%9==0: raise TransientLLMError("503")
        w=set(re.findall(r"[a-z]+",messages[-1]["content"].lower()))
        cat=next((c for c in ["fraud","technical","billing","delivery"] if KW[c]&w),"technical")
        return _r(json.dumps({"category":cat,"reason":"kw"}),messages=messages)
    if "POLICY ASSISTANT" in h:
        u=messages[-1]["content"];m=re.search(r"CONTEXT:\s*(.*?)\s*QUESTION:\s*(.*)",u,re.S)
        ctx,q=m.group(1),m.group(2);qs=set(re.findall(r"[a-z]+",q.lower()))
        sents=[s.strip() for s in re.split(r"(?<=[.])\s+",re.sub(r"\[[^\]]*\]","",ctx))]
        best=max(sents,key=lambda s:len(qs&set(re.findall(r"[a-z]+",s.lower()))))
        return _r(best if qs&set(re.findall(r"[a-z]+",best.lower())) else "I could not find that information in the policy documents.",messages=messages)
    user=next(m["content"] for m in messages if m["role"]=="user");tm=[m for m in messages if m["role"]=="tool"]
    call=lambda n,a:_r(None,[{"id":f"call_{len(tm)+1}","name":n,"arguments":a}],messages)
    if "[LOOP]" in user: return call("search_policy",{"query":"refund policy"})
    if not tm:
        m=re.search(r"ORD-\d{6}",user); return call("get_order",{"order_id":m.group(0)}) if m else _r("DENIED: no order",messages=messages)
    last=tm[-1];d=json.loads(last["content"]);n=last["name"]
    if n=="get_order":
        if "error" in d: return _r("NOT_FOUND: no such order.",messages=messages)
        return call("search_policy",{"query":"refund policy late delivery"})
    o=json.loads(next(m for m in tm if m["name"]=="get_order")["content"])
    if n=="search_policy":
        if o["status"]!="completed": return _r("DENIED: only completed",messages=messages)
        return call("calculate_refund",{"order_amount":o["order_amount"],"delivery_minutes":o["delivery_minutes"]})
    if n=="calculate_refund":
        a=d["refund_inr"]
        if a==0: return _r("DENIED: within 45",messages=messages)
        if a>1000: return call("escalate_to_human",{"order_id":o["order_id"],"refund_inr":a})
        return call("issue_refund",{"order_id":o["order_id"],"refund_inr":a})
    if n=="issue_refund": return _r(f"REFUND: INR {d['refund_inr']} issued.",messages=messages)
    return _r("ESCALATE: sent to human.",messages=messages)
