"""Part A validation: asserts reference checkpoints on the clean files and the SQLite DB."""
import sqlite3
import sys

import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
p = pd.read_csv("data/clean/products.csv")
con = sqlite3.connect("data/instamart.db")
db_counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("customers", "products", "orders")}
done = o[o.status == "completed"]
q5 = pd.read_sql_query(open("sql/queries.sql").read().split("-- Q5")[1].split("-- Q6")[0].split("\n", 1)[1], con)
con.close()

checks = [
    ("customers: 1200 rows", len(c) == 1200),
    ("customers: exactly 6 cities", c.city.nunique() == 6),
    ("customers: gender female/male/unknown = 610/560/30",
     c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("customers: no missing age, all ages 18-60", c.age.isna().sum() == 0 and c.age.between(18, 60).all()),
    ("orders: 12309 rows, one per order_id", len(o) == 12309 and o.order_id.is_unique),
    ("orders: 11110 completed / 1199 cancelled / 0 pending",
     (o.status == "completed").sum() == 11110 and (o.status == "cancelled").sum() == 1199 and (o.status == "pending").sum() == 0),
    ("orders: no missing order_amount", o.order_amount.isna().sum() == 0),
    ("orders: completed have no missing delivery, cancelled all missing",
     done.delivery_minutes.isna().sum() == 0 and o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199),
    ("orders: completed revenue = INR 2,998,668.80", abs(done.order_amount.sum() - 2998668.80) < 0.01),
    ("orders: order_ts in YYYY-MM-DD HH:MM:SS form", o.order_ts.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all()),
    ("db: table row counts match the clean CSVs", db_counts == {"customers": 1200, "products": len(p), "orders": 12309}),
    ("sql Q5: 550 customers above average", len(q5) == 550),
]
failed = [name for name, ok in checks if not ok]
if failed:
    print("FAILED: " + "; ".join(failed), file=sys.stderr)
    sys.exit(1)
print("ALL CHECKPOINTS PASSED")
