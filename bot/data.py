"""Chat data, loaded into memory once (fast: no database query per message) + small JSONL logs."""
import csv
import datetime as dt
import json
import os
import random
import threading

import pandas as pd

PHONES, TICKETS = "data/customer_phones.csv", "outputs/live_tickets.jsonl"
CLAIMS, REORDERS = "outputs/live_claims.jsonl", "outputs/live_orders.jsonl"
REFUNDS = "outputs/live_refunds.jsonl"
_lock = threading.Lock()
_cust = pd.read_csv("data/clean/customers.csv").set_index("customer_id")
_prod = pd.read_csv("data/clean/products.csv").set_index("product_id")
_seg = (pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
        if os.path.exists("outputs/customer_segments.csv") else None)

_df = pd.read_csv("data/clean/orders.csv").join(_prod, on="product_id").sort_values("order_ts", ascending=False)
ORDERS = [{"order_id": r.order_id, "customer_id": r.customer_id, "product": r.product_name, "category": r.category,
           "price": r.price_inr, "qty": int(r.quantity), "ts": r.order_ts, "status": r.status, "amount": float(r.order_amount),
           "minutes": None if pd.isna(r.delivery_minutes) else float(r.delivery_minutes), "discount": r.discount_pct,
           "product_id": r.product_id} for r in _df.itertuples()]
BY_ID = {o["order_id"]: o for o in ORDERS}
BY_CUST = {}
for _o in ORDERS:
    BY_CUST.setdefault(_o["customer_id"], []).append(_o)


def my_orders(cid, limit=None):
    return BY_CUST.get(cid, [])[:limit]


def one_order(oid):
    return BY_ID.get(oid)


def customer_info(cid):
    premium = bool(_seg.loc[cid, "is_premium"]) if _seg is not None and cid in _seg.index else False
    segment = str(_seg.loc[cid, "segment_name"]) if _seg is not None and cid in _seg.index else ""
    return {"customer_id": cid, "city": _cust.loc[cid, "city"], "segment": segment, "premium": premium}


def load_phones():
    """The dataset has no phone numbers, so make demo ones (fixed seed) once. -> {phone: customer_id}"""
    if not os.path.exists(PHONES):
        rng, used = random.Random(2026), set()
        with open(PHONES, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["customer_id", "phone"])
            for cid in _cust.index:
                phone = None
                while phone is None or phone in used:
                    phone = str(rng.randint(6, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))
                used.add(phone)
                w.writerow([cid, phone])
    df = pd.read_csv(PHONES, dtype=str)
    return dict(zip(df.phone, df.customer_id))


def read_log(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def save(path, id_field, prefix, rec):
    """Append one JSON line with a new id like CL-0001; returns the id."""
    with _lock:
        os.makedirs("outputs", exist_ok=True)
        rec = {id_field: f"{prefix}-{len(read_log(path)) + 1:04d}", "ts": dt.datetime.now().isoformat(timespec="seconds"), **rec}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec[id_field]


_HIST = {}                                                          # past tickets per customer (raw export + Part C triage result)
if os.path.exists("data/raw/tickets.csv"):
    _tri = (pd.read_csv("outputs/triage_results.csv", keep_default_na=False).set_index("ticket_id")
            if os.path.exists("outputs/triage_results.csv") else None)
    for _r in pd.read_csv("data/raw/tickets.csv", keep_default_na=False).itertuples(index=False):
        _t = _tri.loc[_r.ticket_id] if _tri is not None and _r.ticket_id in _tri.index else None
        _HIST.setdefault(_r.customer_id, []).append({
            "ticket_id": _r.ticket_id, "order_id": _r.order_id, "text": _r.text, "when": "",
            "category": str(_t["category"]) if _t is not None else "", "status": str(_t["status"]) if _t is not None else "",
            "route": str(_t["route"]) if _t is not None else "human_review", "priority": str(_t["priority"]) if _t is not None else "NORMAL",
            "source": "history"})


def my_tickets(cid, oid=None):
    """Past incident tickets of one customer: raised in this chat (newest first) + the historical ones from Part C."""
    live = [{"ticket_id": r["ticket_id"], "order_id": r.get("order_id", ""), "text": r.get("text", ""), "when": str(r["ts"])[:10], "ts": r["ts"],
             "category": r.get("category", ""), "status": r.get("status", ""), "route": r.get("route", ""),
             "priority": r.get("priority", ""), "source": "chat"}
            for r in read_log(TICKETS)[::-1] if r["customer_id"] == cid]
    rows = live + sorted(_HIST.get(cid, []), key=lambda t: t["ticket_id"], reverse=True)
    return [t for t in rows if oid is None or t["order_id"] == oid]


def my_refunds(cid, oid=None):
    """Refunds raised through the chat (accepted or waiting for approval), newest first."""
    return [r for r in read_log(REFUNDS)[::-1] if r["customer_id"] == cid and (oid is None or r["order_id"] == oid)]


def live_orders(cid):
    """Reorders placed through the chat (kept outside the graded database), newest first."""
    return [{"order_id": r["order_id"], "customer_id": cid, "product_id": r["product_id"], "product": _prod.loc[r["product_id"], "product_name"],
             "category": _prod.loc[r["product_id"], "category"], "price": round(r["amount"] / r["quantity"], 2), "qty": r["quantity"],
             "amount": r["amount"], "status": "placed", "minutes": None, "ts": r["ts"], "discount": 0}
            for r in read_log(REORDERS)[::-1] if r["customer_id"] == cid]
