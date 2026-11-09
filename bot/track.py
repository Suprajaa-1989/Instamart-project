"""Status and timing for refunds, tickets and reorders made in the chat.

Demo extension (not part of the graded brief): a real system would read these from a payments / ticketing service.
Here the stage is worked out from the time since the record was created, on a fast demo clock, so you can watch it advance
(refund: Initiated -> Processing -> Credited in 3 minutes). Set DEMO_SPEED=10 to make it 10x faster.
"""
import datetime as dt
import hashlib
import os

SPEED = float(os.environ.get("DEMO_SPEED", 1))
REFUND_STEPS = [("Initiated", 0), ("Processing with bank / wallet", 60), ("Credited", 180)]
TICKET_STEPS = [("Open", 0), ("In review", 60), ("Resolved", 300)]
ORDER_STEPS = [("Placed", 0), ("Packed", 30), ("Out for delivery", 90), ("Delivered", 180)]
REFUND_ETA_DAYS = (3, 5)                                   # stated to the customer: money back in 3-5 business days (demo assumption)
ORDER_ETA = "about 12 minutes"
SLA = {"HIGH": "2 hours", "NORMAL": "24 hours"}            # first-response target by ticket priority (demo assumption)


def txn(claim_id, order_id):
    """Deterministic demo transaction id for a refund."""
    return "TXN" + hashlib.sha1(f"{claim_id}|{order_id}".encode()).hexdigest()[:10].upper()


def steps(ts, spec):
    """-> (list of {label, state}, current label). state: done / current / todo."""
    elapsed = (dt.datetime.now() - dt.datetime.fromisoformat(ts)).total_seconds() * SPEED
    cur = max(i for i, (_, t) in enumerate(spec) if elapsed >= t)
    last = cur == len(spec) - 1
    return [{"label": lab, "state": "done" if i < cur or (i == cur and last) else "current" if i == cur else "todo"}
            for i, (lab, _) in enumerate(spec)], spec[cur][0]


def fmt(ts):
    return dt.datetime.fromisoformat(ts).strftime("%d %b %Y, %I:%M %p")


def expected_by(ts):
    d = dt.datetime.fromisoformat(ts) + dt.timedelta(days=REFUND_ETA_DAYS[1])
    return d.strftime("%d %b %Y")
