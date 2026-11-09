"""Chat brain.  Buttons -> act()   |   typed text -> handle_text()  (ML intent -> which order -> answer).

A refund / damage CLAIM is an intake form: the bot asks for each fact it needs (reason, delivery timing, units,
photo, unopened, details), keeps them in DRAFT, then policy.decide() checks every rule and explains the result.
Button messages:  menu:<claim|reorder|other|policy|main>  pick:<order>  set:<field>:<value>  review:<order>  re:<order>  reok:<order>
"""
import datetime as dt
import difflib
import re

import rag
import triage_lib
from bot import data, policy, track
from bot.policy import DAYS, REASONS
from intents import LABELS, IntentModel

MODEL, CONF_MIN = IntentModel(), 0.40          # below 40% confidence the bot asks the customer to rephrase
DRAFT, CTX = {}, {}                            # claim being collected / last order discussed, per customer
ROUTE = {"security_team": "our security team", "premium_desk": "your priority premium desk",
         "support_queue": "the support team", "human_review": "a human agent"}
ICON = {"billing": "\U0001F4B3", "delivery": "\U0001F69A", "technical": "\U0001F6E0\uFE0F", "fraud": "\U0001F6E1\uFE0F"}
MENU = [{"label": "\U0001F4B8 Refund / return an item", "send": "menu:claim"}, {"label": "\U0001F501 Reorder", "send": "menu:reorder"},
        {"label": "\U0001F4E6 All my orders", "send": "menu:orders:0"}, {"label": "\U0001F4B0 Track my refunds", "send": "menu:refunds"},
        {"label": "\U0001F3AB My past tickets", "send": "menu:tickets"},
        {"label": "\U0001F4DC Return policy", "send": "menu:policy"}, {"label": "\U0001F4AC Other issue", "send": "menu:other"}]
BACK = {"label": "\u21A9\uFE0F Back to menu", "send": "menu:main"}
YESNO = [{"label": "\u2705 Yes", "value": "yes"}, {"label": "\u274C No", "value": "no"}]
NOT_YOURS = "That order isn't linked to your account, so I can't help with it here."


def R(reply, buttons=None, cards=None, meta=None):
    return {"reply": reply, "buttons": MENU if buttons is None else buttons, "cards": cards or [], "meta": meta}


def item(o):
    return f"{o['product']} x{o['qty']}"


def refund_by_order(cid):
    out = {}
    for r in reversed(data.my_refunds(cid)):                      # oldest first so the newest refund wins
        out[r["order_id"]] = "Awaiting approval" if r["needs_approval"] else track.steps(r["ts"], track.REFUND_STEPS)[1]
    return out


def orders_card(rows, title="Your recent orders"):
    rf = refund_by_order(rows[0]["customer_id"]) if rows else {}
    return {"type": "orders", "title": title, "items": [{"order_id": o["order_id"], "product": o["product"], "qty": o["qty"], "amount": round(o["amount"], 2),
            "category": o.get("category", ""), "refund": rf.get(o["order_id"], ""), "status": o["status"], "minutes": o["minutes"], "date": str(o["ts"])[:10]} for o in rows]}


def order_state(o):
    return (f"was delivered on {str(o['ts'])[:10]}" + (f" in {o['minutes']:.0f} minutes" if o["minutes"] is not None else "")) if o["status"] == "completed" else f"is {o['status']}"


def detail_card(o):
    return {"type": "detail", "order_id": o["order_id"], "product": o["product"], "category": o.get("category", ""), "qty": o["qty"], "unit": o["price"],
            "discount": o["discount"], "amount": round(o["amount"], 2), "status": o["status"], "date": str(o["ts"])[:16], "minutes": o["minutes"]}


def refund_card(r):
    """Refund tracker: amount, reference + transaction id, timing and a status timeline."""
    inr = f"INR {r['amount']:,.2f}"
    if r["needs_approval"]:
        steps = [{"label": "Claim submitted", "state": "done"}, {"label": "Human lead approval", "state": "current"},
                 {"label": "Refund initiated", "state": "todo"}, {"label": "Credited", "state": "todo"}]
        status, txn, eta = "Awaiting approval", "assigned after approval", "after a human lead approves (refunds above INR 1000)"
    else:
        steps, status = track.steps(r["ts"], track.REFUND_STEPS)
        txn, eta = r["txn_ref"], f"within {track.REFUND_ETA_DAYS[0]}-{track.REFUND_ETA_DAYS[1]} business days (by {track.expected_by(r['ts'])})"
    rows = [["Status", status], ["Refund amount", inr, True], ["Refund to", "Original payment method"], ["Initiated", track.fmt(r["ts"])],
            ["Expected credit", eta], ["Reference ID", r["refund_id"]], ["Transaction ID", txn], ["Claim", r["claim_id"]], ["Order", f"{r['order_id']} - {r['item']}"]]
    return {"type": "timeline", "title": f"\U0001F4B0 Refund {r['refund_id']} \u00B7 {status}", "rows": rows, "steps": steps,
            "state": "ok" if status == "Credited" else "warn" if r["needs_approval"] else "",
            "note": "Demo clock: the status advances every minute so you can watch it change."}


def order_track_card(o):
    steps, status = track.steps(o["ts"], track.ORDER_STEPS)
    return {"type": "timeline", "title": f"\U0001F6F5 Order {o['order_id']} \u00B7 {status}", "state": "ok" if status == "Delivered" else "", "steps": steps,
            "rows": [["Status", status], ["Placed", track.fmt(o["ts"])], ["Estimated delivery", track.ORDER_ETA], ["Total", f"INR {o['amount']:,.2f}", True]]}


def claim_status(c, refunds):
    if c["outcome"] == "rejected":
        return "Rejected"
    r = next((x for x in refunds if x["claim_id"] == c["claim_id"]), None)
    return "Awaiting approval" if r is None or r["needs_approval"] else track.steps(r["ts"], track.REFUND_STEPS)[1]


def refunds_view(cid, oid=None):
    rows, scope = data.my_refunds(cid, oid), f" on order {oid}" if oid else ""
    if not rows:
        return R(f"You have no refunds{scope} yet. Tap \u201CRefund / return an item\u201D to start a claim.")
    return R(f"You have {len(rows)} refund(s){scope}. Latest: {rows[0]['refund_id']} for INR {rows[0]['amount']:,.2f}.", cards=[refund_card(r) for r in rows[:5]])


PER_PAGE = 8


def orders_view(cid, page=0):
    """Every order of the customer, 8 per page, with a 'show more' button (live reorders first)."""
    rows = data.live_orders(cid) + data.my_orders(cid)
    if not rows:
        return R("You have no orders yet.")
    chunk = rows[page * PER_PAGE:(page + 1) * PER_PAGE]
    if not chunk:
        return R("That's all of your orders.")
    start, end = page * PER_PAGE + 1, page * PER_PAGE + len(chunk)
    more = [{"label": f"\u25B8 Show more orders ({len(rows) - end} left)", "send": f"menu:orders:{page + 1}"}] if end < len(rows) else []
    return R(f"Orders {start}-{end} of {len(rows)}. Tap any order to see the item details and what you can do with it.",
             cards=[orders_card(chunk, f"Your orders ({start}-{end} of {len(rows)})")], buttons=more + MENU)


def tickets_view(cid, oid=None):
    """Past incident tickets (and claims) of this customer, optionally only for one order."""
    rows, scope = data.my_tickets(cid, oid), f" on order {oid}" if oid else ""
    claims = [c for c in data.read_log(data.CLAIMS) if c["customer_id"] == cid and (oid is None or c["order_id"] == oid)][::-1]
    if not rows and not claims:
        return R(f"You have no past tickets{scope}. If something is wrong, tap \u201COther issue\u201D and I'll raise one.")
    items = [{"ticket_id": t["ticket_id"], "order_id": t["order_id"], "when": t["when"], "icon": ICON.get(t["category"], "\U0001F4DD"),
              "category": t["category"] or "needs review", "route": ROUTE.get(t["route"], t["route"] or "a human agent"), "priority": t["priority"], "ref": t["ticket_id"],
              "status": "Closed (past ticket)" if t["source"] == "history" else track.steps(t["ts"], track.TICKET_STEPS)[1],
              "sla": track.SLA.get(t["priority"], "") if t["source"] != "history" else "",
              "text": (lambda x: "(no message text)" if not x else x if len(x) <= 140 else x[:137] + "...")(triage_lib.redact(t["text"]).strip())}
             for t in rows[:10]]
    cards = [{"type": "tickets", "title": f"Past tickets{scope} ({len(rows)})", "items": items, "total": len(rows)}] if rows else []
    if claims:
        cards.append({"type": "claims", "items": [{"claim_id": c["claim_id"], "item": c["item"], "outcome": c["outcome"], "refund": c["refund"], "when": str(c["ts"])[:10],
                                         "status": claim_status(c, data.my_refunds(cid))} for c in claims[:10]]})
    return R(f"You have {len(rows)} past ticket(s) and {len(claims)} claim(s){scope}.", cards=cards)


def spend_card(cid):
    rows = data.my_orders(cid)
    done = [o for o in rows if o["status"] == "completed"]
    if not done:
        return None
    qty = {}
    for o in done:
        qty[o["product"]] = qty.get(o["product"], 0) + o["qty"]
    top, total = max(qty, key=qty.get), sum(o["amount"] for o in done)
    return {"type": "summary", "completed": len(done), "cancelled": len(rows) - len(done), "total": round(total, 2),
            "average": round(total / len(done), 2), "favourite": f"{top} (x{qty[top]})", "last": str(done[0]["ts"])[:10]}


def raise_ticket(cid, text, oid="", kind="complaint", category=None, it="", override=None):
    """-> (text, ticket card). Category from the ML model/button, else the Part C classifier; routing = Part C rules."""
    status, category = (override, category or "") if override else ("classified_json", category) if category else triage_lib.classify(text)
    route, priority = triage_lib.route(status, category, data.customer_info(cid)["premium"])
    tid = data.save(data.TICKETS, "ticket_id", "LT", {"customer_id": cid, "order_id": oid, "item": it, "kind": kind, "text": triage_lib.redact(text),
                    "status": status, "category": category, "route": route, "priority": priority})
    card = {"type": "ticket", "ticket_id": tid, "category": category or "needs review", "icon": ICON.get(category, "\U0001F4DD"),
            "item": it, "route": ROUTE[route], "priority": priority, "status": "Open", "sla": track.SLA.get(priority, "24 hours"),
            "created": track.fmt(dt.datetime.now().isoformat(timespec="seconds")),
            "steps": track.steps(dt.datetime.now().isoformat(timespec="seconds"), track.TICKET_STEPS)[0]}
    return f"I'm sorry about that. I've raised ticket {tid}" + (f" ({category})" if category else "") + f" and routed it to {ROUTE[route]} with {priority} priority.", card


# ------------------------------------------------------------------ claim intake
def needed(o, d):
    """The facts still required, in the order the bot asks for them."""
    r = d.get("reason")
    fields = ["reason", "days"]
    if r and r != "late":
        fields += ["units"] + (["photo"] if r in policy.PHOTO_REASONS else []) + (["unopened"] if r == "wrong" and o["category"] in policy.HYGIENE else []) + ["note"]
    return fields


def form_card(o, d):
    fmt = {"reason": lambda v: REASONS[v][1], "days": lambda v: DAYS[v], "photo": str.title, "unopened": str.title}
    names = {"reason": "Reason", "days": "Delivered", "units": "Units affected", "photo": "Photo evidence", "unopened": "Item unopened", "note": "Details"}
    rows = [["Order", o["order_id"]], ["Item", item(o)]] + [[names[k], fmt.get(k, str)(d[k])] for k in names if k in d]
    return {"type": "form", "rows": rows}


def ask(cid):
    """Ask the next missing question; when everything is collected, decide."""
    d = DRAFT[cid]
    o = data.one_order(d["oid"])
    for n, field in enumerate(needed(o, d), 1):
        if field == "units" and field not in d and o["qty"] == 1:
            d["units"] = "1"
        if field in d:
            continue
        step = f"Step {n}: "
        opts = {"reason": [{"label": f"{i} {t}", "value": k} for k, (i, t) in REASONS.items()],
                "days": [{"label": t, "value": k} for k, t in DAYS.items()],
                "units": [{"label": str(u), "value": str(u)} for u in range(1, min(o["qty"], 6) + 1)],
                "photo": YESNO, "unopened": YESNO, "note": [{"label": "Skip", "value": "-"}]}[field]
        text = {"reason": f"You ordered {item(o)}. What went wrong?", "days": "When did you receive this order?",
                "units": f"How many units of {o['product']} are affected?", "photo": "Do you have a photo of the problem? (Our team will ask for it after you submit.)",
                "unopened": "Is the item unopened and unused?", "note": "Briefly describe what happened (type your message), or skip."}[field]
        d["await"] = field == "note"
        return R(step + text, cards=[form_card(o, d)], buttons=[{"label": b["label"], "send": f"set:{field}:{b['value']}"} for b in opts] + [BACK])
    return finish(cid)


def finish(cid):
    d = DRAFT.pop(cid)
    o = data.one_order(d["oid"])
    dec = policy.decide(o, d, [c for c in data.read_log(data.CLAIMS) if c["order_id"] == o["order_id"]])
    answers = {k: v for k, v in d.items() if k not in ("oid", "reason", "await")}
    cl = data.save(data.CLAIMS, "claim_id", "CL", {"customer_id": cid, "order_id": o["order_id"], "item": item(o), "reason": d["reason"],
                   "answers": answers, "outcome": dec["outcome"], "refund": dec["refund"], "explanation": dec["explanation"]})
    source, text = policy.reference(d["reason"])
    cards = [{"type": "decision", "claim_id": cl, "item": item(o), "reason": REASONS[d["reason"]][1], "source": source, "policy": text, **dec}]
    buttons = [{"label": "\U0001F469\u200D\U0001F4BC Ask a human to review", "send": f"review:{o['order_id']}"}, BACK] if dec["outcome"] == "rejected" else None
    if dec["outcome"] == "escalated":
        t_text, t_card = raise_ticket(cid, f"Refund of INR {dec['refund']:,.2f} for {item(o)} needs approval", o["order_id"], "claim_escalation", None, item(o), "human_requested")
        cards.append(t_card)
    text = dec["explanation"]
    if dec["outcome"] in ("accepted", "escalated") and dec["refund"] > 0:
        esc = dec["outcome"] == "escalated"
        rid = data.save(data.REFUNDS, "refund_id", "RF", {"customer_id": cid, "claim_id": cl, "order_id": o["order_id"], "item": item(o), "amount": dec["refund"],
                        "needs_approval": esc, "txn_ref": "" if esc else track.txn(cl, o["order_id"]), "method": "Original payment method"})
        cards.append(refund_card(data.my_refunds(cid)[0]))
        text += (f" Refund reference {rid} is awaiting approval by a human lead." if esc else
                 f" Refund {rid} (transaction {track.txn(cl, o['order_id'])}) is initiated and should reach you within {track.REFUND_ETA_DAYS[0]}-{track.REFUND_ETA_DAYS[1]} business days.")
    return R(text, cards=cards, buttons=buttons)


# ------------------------------------------------------------------ button flows
def menu(cid, kind, page=0):
    if kind == "orders":
        return orders_view(cid, page)
    if kind == "tickets":
        return tickets_view(cid)
    if kind == "refunds":
        return refunds_view(cid)
    if kind == "policy":
        return R("Here is our return and refund policy.", cards=[{"type": "policy", "title": "Return & refund policy", "text": policy.policy_text()}])
    if kind == "other":
        DRAFT[cid] = {"oid": "", "await": "other"}
        return R("Please describe the problem in your own words and I'll raise a ticket.", buttons=[BACK])
    if kind == "main":
        DRAFT.pop(cid, None)
        return R("Okay. What else can I help you with?")
    rows = [o for o in data.my_orders(cid) if kind == "claim" or o["status"] == "completed"]       # all orders, 6 buttons per page
    prefix, chunk = "pick" if kind == "claim" else "re", rows[page * 6:(page + 1) * 6]
    buttons = [{"label": f"{o['order_id']} \u00B7 {o['product']} \u00B7 INR {o['amount']:,.2f}", "send": f"{prefix}:{o['order_id']}"} for o in chunk]
    if (page + 1) * 6 < len(rows):
        buttons.append({"label": f"\u25B8 More orders ({len(rows) - (page + 1) * 6} left)", "send": f"menu:{kind}:{page + 1}"})
    ask_text = "Which order is this about?" if kind == "claim" else "Which order would you like to place again?"
    return R(ask_text, buttons=buttons + [BACK]) if buttons else R("I couldn't find a suitable recent order on your account.")


def act(cid, msg):
    kind, *a = msg.split(":")
    if kind == "menu":
        return menu(cid, a[0], int(a[1]) if len(a) > 1 else 0)
    if kind == "tix":                                             # tickets / refunds are always filtered by the signed-in customer
        return tickets_view(cid, a[0])
    if kind == "rf":
        return refunds_view(cid, a[0])
    if kind == "ord" and a[0].startswith("LIVE-"):                # reorder placed in this chat
        o = next((x for x in data.live_orders(cid) if x["order_id"] == a[0]), None)
        if o is None:
            return R(NOT_YOURS)
        return R(f"{item(o)} - order {o['order_id']} was placed in this chat.", cards=[detail_card(o), order_track_card(o)],
                 buttons=[{"label": "\U0001F3AB Tickets on this order", "send": f"tix:{o['order_id']}"}, {"label": "\U0001F4E6 All my orders", "send": "menu:orders:0"}, BACK])
    if kind == "set":                                             # answer to an intake question
        if cid not in DRAFT:
            return R("That claim has expired. Please start again from the menu.")
        value = ":".join(a[1:])
        DRAFT[cid][a[0]] = "No details given" if (a[0], value) == ("note", "-") else value
        return ask(cid)
    o = data.one_order(a[0])
    if o is None or o["customer_id"] != cid:
        return R(NOT_YOURS)
    CTX[cid] = o["order_id"]
    if kind == "ord":                                             # customer tapped an order / item in a card
        oid = o["order_id"]
        acts = [{"label": "\U0001F4B8 Refund / return this item", "send": f"pick:{oid}"}]
        if o["status"] == "completed":
            acts.append({"label": "\U0001F501 Reorder this item", "send": f"re:{oid}"})
        has_refund = data.my_refunds(cid, oid)
        if has_refund:
            acts.append({"label": "\U0001F4B0 Refund status", "send": f"rf:{oid}"})
        acts += [{"label": "\U0001F3AB Tickets on this order", "send": f"tix:{oid}"}, {"label": "\U0001F4E6 All my orders", "send": "menu:orders:0"}, BACK]
        return R(f"{item(o)} ({o['category']}) - order {oid} {order_state(o)}. What would you like to do with it?",
                 cards=[detail_card(o)] + ([refund_card(has_refund[0])] if has_refund else []), buttons=acts)
    if kind == "pick":
        DRAFT[cid] = {"oid": o["order_id"]}
        return ask(cid)
    if kind == "review":
        t_text, card = raise_ticket(cid, f"Please review the rejected claim for {item(o)}", o["order_id"], "claim_review", None, item(o), "human_requested")
        return R(f"Done. A human agent will review it. {t_text}", cards=[card])
    total = round(o["price"] * o["qty"], 2)
    card = {"type": "reorder", "product": o["product"], "qty": o["qty"], "unit": o["price"], "total": total, "state": "preview"}
    if kind == "re":
        return R(f"Place this order again? {item(o)} = INR {total:,.2f} (demo: no payment is taken).", cards=[card],
                 buttons=[{"label": f"\u2705 Confirm reorder (INR {total:,.2f})", "send": f"reok:{o['order_id']}"}, {"label": "\u274C Cancel", "send": "menu:main"}])
    new_id = data.save(data.REORDERS, "order_id", "LIVE", {"customer_id": cid, "product_id": o["product_id"], "quantity": o["qty"],
                                                          "amount": total, "status": "placed", "reorder_of": o["order_id"]})
    return R(f"Done! Order {new_id} placed: {item(o)}. Estimated delivery {track.ORDER_ETA}.",
             cards=[{**card, "state": "placed", "order_id": new_id}, order_track_card(data.live_orders(cid)[0])])


# ------------------------------------------------------------------ typed text
GENERIC = {"item", "items", "pack", "product", "order", "kg", "ml"}


def resolve_order(cid, msg, explicit_only):
    """Which order is meant? id > item name > 'it/that' > first/last > (latest, for info questions only)."""
    low, m = msg.lower(), re.search(r"ORD-\d{6}", msg)
    if m:
        return data.one_order(m.group(0)), "id"
    rows, words = data.my_orders(cid), set(re.findall(r"[a-z]+", msg.lower()))
    best, score = None, 0.0
    for o in rows:                                                 # item mentioned? ("the milk")
        toks = [w for w in re.findall(r"[a-z]+", o["product"].lower()) if len(w) >= 3 and w not in GENERIC]
        hits = sum(1 for w in toks if w in words or difflib.get_close_matches(w, words, n=1, cutoff=0.82))
        if toks and hits / len(toks) >= 0.5 and hits / len(toks) > score:
            best, score = o, hits / len(toks)
    if best:
        return best, "item"
    if CTX.get(cid) and re.search(r"\b(it|that|this|same)\b", low):
        return data.one_order(CTX[cid]), "context"
    if rows and re.search(r"\b(first|oldest|earliest)\b", low):
        return rows[-1], "ordinal"
    if rows and re.search(r"\b(last|latest|recent|previous|newest)\b", low):
        return rows[0], "ordinal"
    return (rows[0], "default") if rows and not explicit_only else (None, "none")


def reason_from(low):
    for key, words in (("wrong", ("wrong",)), ("missing", ("missing",)), ("quality", ("expired", "spoil", "rotten", "stale"))):
        if any(w in low for w in words):
            return key
    return "damaged"


def handle_text(cid, message):
    msg = message.strip()
    if not msg:
        return R("Please type your message.")
    d = DRAFT.get(cid)
    if d and d.get("await"):                                       # the customer is answering a free-text question
        if d["await"] == "other":
            DRAFT.pop(cid)
            text, card = raise_ticket(cid, msg, CTX.get(cid, ""))
            return R(text, cards=[card])
        d["note"], d["await"] = msg, False
        return ask(cid)
    low0 = msg.lower()
    if re.search(r"\brefund\b", low0) and re.search(r"\b(status|track|where|when|received|credited|reference|transaction)\b", low0):
        return refunds_view(cid, (re.search(r"ORD-\d{6}", msg) or [None])[0])
    if re.search(r"\b(past|previous|earlier|old|my|all)\b.*\b(tickets?|complaints?|incidents?)\b", msg.lower()):
        return tickets_view(cid, (re.search(r"ORD-\d{6}", msg) or [None])[0])
    intent, conf, _ = MODEL.predict(msg)
    meta, low, has_id = {"intent": LABELS[intent], "confidence": round(conf * 100)}, msg.lower(), re.search(r"ORD-\d{6}", msg)
    if conf < CONF_MIN:
        if not has_id:
            return R("I'm not fully sure I understood that. Could you rephrase it, or pick one of the options below?", meta=meta)
        intent = "order_details"

    def done(res):
        res["meta"] = meta
        return res

    if intent in ("greeting", "thanks"):
        return done(R("Hello! How can I help you today?" if intent == "greeting" else "You're welcome! Anything else I can help with?"))
    if intent == "policy":
        return done(menu(cid, "policy") if "refund" in low or "return" in low else R(rag.answer(msg)[0]))
    if intent == "order_history":
        return done(orders_view(cid, 0))
    if intent == "spend_summary":
        card = spend_card(cid)
        return done(R("Here is your spending summary." if card else "You have no completed orders yet.", cards=[card] if card else []))

    claim = intent in ("refund", "late_delivery", "damaged", "reorder")
    o, how = resolve_order(cid, msg, explicit_only=claim or intent in ("billing", "technical", "fraud", "human"))
    if has_id and o is None:
        return done(R(f"I couldn't find order {has_id.group(0)}. Please check the order id."))
    if o and o["customer_id"] != cid:
        return done(R(NOT_YOURS))
    if intent in ("billing", "technical", "fraud", "human"):
        human = intent == "human"
        text, card = raise_ticket(cid, msg, o["order_id"] if o else "", intent, None if human else intent, item(o) if o else "", "human_requested" if human else None)
        return done(R(f"Sure, a human agent will take over. {text}" if human else text, cards=[card]))
    if claim:
        if o is None:                                              # money is involved: never guess the order
            return done(menu(cid, "reorder" if intent == "reorder" else "claim"))
        if intent == "reorder":
            return done(act(cid, f"re:{o['order_id']}"))
        DRAFT[cid] = {"oid": o["order_id"], **({"reason": "late"} if intent == "late_delivery" else {"reason": reason_from(low)} if intent == "damaged" else {})}
        CTX[cid] = o["order_id"]
        return done(ask(cid))
    if o is None:
        return done(R("I couldn't find any orders on your account yet."))
    CTX[cid] = o["order_id"]
    state = order_state(o)
    lead = f"Your order {o['order_id']} ({item(o)}) {state}." if intent == "order_status" else f"Here are the details of {o['order_id']}" + (" (your most recent order)." if how == "default" else ".")
    return done(R(lead, cards=[detail_card(o)]))


def dispatch(cid, msg):
    return act(cid, msg) if re.match(r"^(menu|pick|set|review|re|reok|ord|tix|rf):", msg) else handle_text(cid, msg)
