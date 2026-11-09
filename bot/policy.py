"""Return / refund policy engine: facts collected from the customer -> every rule checked -> decision with reasons.
Rules 1, 3 and 9 come from the brief (via agent.calculate_refund); the others are this project's demo policy extension."""
import rag
from agent import TOOL_MAP

REASONS = {"late": ("\u23F1\uFE0F", "Delivered late"), "damaged": ("\U0001F4E6", "Item damaged"), "wrong": ("\u274C", "Wrong item delivered"),
           "missing": ("\U0001F9E9", "Item missing"), "quality": ("\U0001F940", "Spoiled / expired"), "other": ("\U0001F4AC", "Other reason")}
PHOTO_REASONS, HYGIENE = ("damaged", "wrong", "quality"), ("Personal Care", "Electronics")
DAYS = {"0": "Today", "3": "1-3 days ago", "7": "4-7 days ago", "8": "More than 7 days ago"}
RULES_TEXT = [
    "1. Only delivered (completed) orders can be claimed.",
    "2. A claim must be raised within 7 days of delivery.",
    "3. Late delivery: more than 45 minutes after ordering = full refund; 45 minutes or less = no refund.",
    "4. Damaged, wrong or spoiled items need photo evidence.",
    "5. Wrong Personal Care or Electronics items must be unopened and unused.",
    "6. Spoiled or expired dairy must be reported within 3 days.",
    "7. Damaged, wrong, missing or spoiled items are refunded for the units affected only.",
    "8. Only one accepted claim per order and reason.",
    "9. Refunds above INR 1000 need a human lead's approval."]


def decide(o, d, earlier):
    """o = order, d = answers collected from the customer, earlier = claims already made on this order.
    -> {outcome: accepted|escalated|rejected, refund, rules: [{rule, ok, why}], explanation}"""
    reason, rules = d["reason"], []

    def rule(name, ok, passed, failed):
        rules.append({"rule": name, "ok": ok, "why": passed if ok else failed})

    delivered = o["status"] == "completed"
    rule("Order was delivered", delivered, "the order was delivered", f"the order is {o['status']}, so nothing was delivered")
    rule("Claimed within 7 days of delivery", d["days"] != "8", "you are inside the 7-day window", "the 7-day claim window has passed")
    rule("No earlier accepted claim for this reason", not any(c["reason"] == reason and c["outcome"] != "rejected" for c in earlier),
         "there is no earlier claim for this reason", "a claim for this reason was already accepted")
    if reason == "late":
        m = o["minutes"]
        refund = TOOL_MAP["calculate_refund"](o["amount"], m)["refund_inr"] if delivered else 0.0
        rule("Delivered more than 45 minutes after ordering", refund > 0, f"it took {m:.0f} minutes, more than the 45-minute promise",
             "it arrived within the 45-minute promise" if m is not None else "there is no delivery time on record")
    else:
        refund = round(o["amount"] / o["qty"] * int(d.get("units", 1)), 2)
        if reason in PHOTO_REASONS:
            rule("Photo evidence available", d.get("photo") == "yes", "you have photo evidence", "photo evidence is required for this reason")
        if reason == "wrong" and o["category"] in HYGIENE:
            rule("Item is unopened and unused", d.get("unopened") == "yes", "the item is unopened", f"{o['category']} items must be unopened to be returned")
        if reason == "quality" and o["category"] == "Dairy":
            rule("Spoiled dairy reported within 3 days", d["days"] in ("0", "3"), "it was reported in time", "spoiled dairy must be reported within 3 days")
    ok = all(r["ok"] for r in rules)
    outcome = "rejected" if not ok else "escalated" if refund > 1000 else "accepted"
    if outcome == "rejected":
        refund, text = 0.0, "Rejected because " + "; ".join(r["why"] for r in rules if not r["ok"]) + "."
    elif outcome == "escalated":
        text = f"All {len(rules)} policy rules are met, but INR {refund:,.2f} is above INR 1000, so a human lead must approve it."
    else:
        text = f"Accepted: all {len(rules)} policy rules are met, so a refund of INR {refund:,.2f} is approved."
    return {"outcome": outcome, "refund": refund, "rules": rules, "explanation": text}


def reference(reason):
    """The policy sentence from data/policies that backs the decision: (file, text)."""
    hit = rag.retrieve(f"refund policy {REASONS[reason][1]}", k=1)[0]
    return hit["source"], hit["text"]


def policy_text():
    return "\n".join(RULES_TEXT) + "\n\nFrom the policy document:\n" + open("data/policies/refund_policy.txt", encoding="utf-8").read().strip()
