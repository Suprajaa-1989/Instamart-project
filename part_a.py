import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd

from util import md_table, write

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
OUT, REP = "outputs", "outputs/readme_parts"
for d in (CLEAN, OUT, REP, "sql"):
    os.makedirs(d, exist_ok=True)

customers = pd.read_csv(f"{RAW}/customers.csv")
products = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")
tickets = pd.read_csv(f"{RAW}/tickets.csv", keep_default_na=False)

# ----------------------------------------------------------------------------
# Task 2 - audit of the raw data
# ----------------------------------------------------------------------------


def missing_table(df):
    blanks = df.isna().sum() + df.eq("").sum()
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": blanks.values,
        "missing_pct": (blanks / len(df) * 100).round(2).values,
    })


amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
dl_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(customers["age"], errors="coerce")
audit = {
    "orders raw rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "rows with status 'pending'": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "order_amount not convertible to a number": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((dl_num < 0).sum()),
    "blank delivery_minutes": int(dl_num.isna().sum()),
    "customers: blank age": int(age_num.isna().sum()),
    "customers: age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "customers: distinct raw city spellings": int(customers["city"].nunique()),
    "customers: distinct non-blank raw gender spellings": int(customers["gender"].nunique()),
    "customers: blank gender": int(customers["gender"].isna().sum()),
}
audit_df = pd.DataFrame({"audit figure": list(audit), "value": list(audit.values())})

shape_rows = [{"table": n, "rows": d.shape[0], "columns": d.shape[1]}
              for n, d in [("customers", customers), ("products", products),
                           ("orders", orders), ("tickets", tickets)]]

# ----------------------------------------------------------------------------
# Task 3 - clean customers
# ----------------------------------------------------------------------------
log = {}
c = customers.copy()
c["city"] = c["city"].astype(str).str.strip().str.title()
log["customers: city spellings -> clean cities"] = f"{audit['customers: distinct raw city spellings']} -> {c['city'].nunique()}"

g = c["gender"].fillna("").astype(str).str.strip().str.lower()
c["gender"] = g.map({"male": "male", "m": "male", "female": "female", "f": "female"}).fillna("unknown")
log["customers: gender values mapped to male/female/unknown"] = f"{audit['customers: distinct non-blank raw gender spellings']} spellings + {audit['customers: blank gender']} blanks -> 3 values"

age = pd.to_numeric(c["age"], errors="coerce")
n_blank_age = int(age.isna().sum())
bad_age = int((age.notna() & ~age.between(10, 100)).sum())
age = age.where(age.between(10, 100))                         # impossible ages -> missing
age = age.fillna(age.groupby(c["city"]).transform("median"))  # median age of the customer's city
c["age"] = age
log["customers: blank ages imputed with city median"] = n_blank_age
log["customers: ages outside 10-100 set to missing, then imputed"] = bad_age
c["signup_date"] = pd.to_datetime(c["signup_date"])

# ----------------------------------------------------------------------------
# Task 4 - clean orders (rules i-v, in the stated order)
# ----------------------------------------------------------------------------
o = orders.copy()
o["order_ts"] = pd.to_datetime(o["order_ts"])                              # (i)
n0 = len(o)
o = o.drop_duplicates()                                                    # (ii)
log["orders: exact duplicate rows removed"] = n0 - len(o)
n1 = len(o)
o = o.sort_values(["order_ts", "order_id"]).drop_duplicates("order_id", keep="last")  # (iii)
log["orders: superseded (earlier 'pending') records removed, latest order_ts kept"] = n1 - len(o)

o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")      # (iv)
miss_amt = o["order_amount"].isna()
calc = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o.loc[miss_amt, "order_amount"] = calc[miss_amt]
log["orders: order_amount unparseable -> recomputed from price x qty x (1-disc)"] = int(miss_amt.sum())

o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # (v)
neg = o["delivery_minutes"] < 0
log["orders: negative delivery_minutes set to missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
done = o["status"] == "completed"
cat_median = o.loc[done].groupby("category")["delivery_minutes"].median()
fill = done & o["delivery_minutes"].isna()
o.loc[fill, "delivery_minutes"] = o.loc[fill, "category"].map(cat_median)
log["orders: completed orders' missing delivery_minutes -> category median (completed)"] = int(fill.sum())
log["orders: cancelled orders' delivery_minutes left missing"] = int(((o["status"] == "cancelled") & o["delivery_minutes"].isna()).sum())

o = o.sort_values("order_ts").reset_index(drop=True)
o = o[["order_id", "customer_id", "product_id", "quantity", "order_ts",
       "status", "discount_pct", "order_amount", "delivery_minutes"]]

# ----------------------------------------------------------------------------
# Task 5 - store clean tables as CSV and in SQLite (timestamps as YYYY-MM-DD HH:MM:SS text)
# ----------------------------------------------------------------------------
c_out = c.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
o_out = o.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
products.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)

if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
c_out.to_sql("customers", con, index=False)
products.to_sql("products", con, index=False)
o_out.to_sql("orders", con, index=False)
con.commit()

# ----------------------------------------------------------------------------
# Task 6 - SQL analysis (queries live in sql/queries.sql)
# ----------------------------------------------------------------------------
sql_text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d+)[^\n]*\n", sql_text, flags=re.M)
queries = dict(zip(parts[1::2], parts[2::2]))
res = {q: pd.read_sql_query(s, con) for q, s in queries.items()}
con.close()

q6 = res["Q6"]
tie_rows = int((q6["rnk"] != q6["row_num"]).sum())
top_city = q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].reset_index(drop=True)
q2 = res["Q2"]
sql_summary = pd.DataFrame([
    ["Q1 WHERE then HAVING", len(res["Q1"]), "30 customers"],
    ["Q2 three-table join", len(q2), f"30 rows; largest = {q2.iloc[0]['city']} / {q2.iloc[0]['category']} INR {q2.iloc[0]['revenue']:,.2f} (ref. Bengaluru / Electronics 303,909.45)"],
    ["Q3 scalar subquery", len(res["Q3"]), "11 products"],
    ["Q4 LEFT JOIN", len(res["Q4"]), "25 customers"],
    ["Q5 CTE", len(res["Q5"]), f"550 customers; average INR {res['Q5']['avg_spend'].iloc[0]:,.2f} (ref. 2,552.06)"],
    ["Q6 window functions", len(q6), f"{tie_rows} rows where RANK() differs from ROW_NUMBER() (ref. 8)"],
    ["Q7 running total", len(res["Q7"]), f"183 daily rows; last cumulative INR {res['Q7']['cumulative_revenue'].iloc[-1]:,.2f} (ref. 2,998,668.80)"],
], columns=["query", "rows returned", "reference / key result"])

# ----------------------------------------------------------------------------
# Task 7 - pandas cross-check + EDA
# ----------------------------------------------------------------------------
cd = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                  .merge(products[["product_id", "category"]], on="product_id")
pivot = pd.pivot_table(cd, index="city", columns="category", values="order_amount",
                       aggfunc="sum", margins=True, margins_name="Total").round(2)
grand_total = float(pivot.loc["Total", "Total"])
assert abs(grand_total - float(cd["order_amount"].sum())) < 0.01
assert abs(grand_total - float(res["Q2"]["revenue"].sum())) < 0.05, "pandas pivot must equal SQL Q2"
by_city = cd.groupby("city").agg(order_sum=("order_amount", "sum"),
                                 order_mean=("order_amount", "mean"),
                                 order_count=("order_amount", "count")).round(2)

x = cd["order_amount"]
q1_, q3_ = x.quantile(0.25), x.quantile(0.75)
iqr = q3_ - q1_
lo_f, hi_f = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
n_hi, n_lo = int((x > hi_f).sum()), int((x < lo_f).sum())
stats = {"mean": x.mean(), "median": x.median(), "skewness": x.skew(), "Q1": q1_, "Q3": q3_,
         "lower fence": lo_f, "upper fence": hi_f, "outliers above": n_hi, "outliers below": n_lo}
corr = cd[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
pairs.sort(key=lambda t: -abs(t[2]))
top_pair = pairs[0]

eda_text = (
    f"**What is this data?** {len(cd):,} completed Instamart orders from {cd['order_ts'].min():%d %b %Y} to "
    f"{cd['order_ts'].max():%d %b %Y}, with amount (INR), quantity, discount and delivery time. "
    f"**Distribution:** order_amount is right-skewed (mean {x.mean():.2f} vs median {x.median():.2f}, skewness {x.skew():.2f}); "
    f"half of orders lie between {q1_:.2f} and {q3_:.2f} at the quartiles. "
    f"**Outliers:** {n_hi} orders exceed the upper 1.5xIQR fence of {hi_f:.2f} and {n_lo} fall below the lower fence; "
    f"they look like genuine high-priced baskets (likely electronics), not errors. "
    f"**Relationships:** the largest absolute correlation is {top_pair[0]} with {top_pair[1]} "
    f"(r = {top_pair[2]:+.3f}); r of this size is weak and does not establish causation."
)
n_words = len(eda_text.replace("*", "").split())
assert n_words <= 200, f"EDA answer has {n_words} words (limit 200)"

# ----------------------------------------------------------------------------
# Persist results + README section
# ----------------------------------------------------------------------------
json.dump({"audit": audit, "cleaning_log": {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in log.items()},
           "completed_amount_sum": round(float(o.loc[o.status == "completed", "order_amount"].sum()), 2)},
          open(f"{OUT}/part_a_results.json", "w"), indent=2, default=str)

n_comp = int((o["status"] == "completed").sum())
n_canc = int((o["status"] == "cancelled").sum())
clean_facts = pd.DataFrame([
    ["orders rows (one per order_id)", len(o), "12,309"],
    ["completed / cancelled", f"{n_comp:,} / {n_canc:,}", "11,110 / 1,199"],
    ["pending rows", int((o.status == 'pending').sum()), "0"],
    ["missing order_amount", int(o.order_amount.isna().sum()), "0"],
    ["missing delivery_minutes (completed / cancelled)",
     f"{int(o.loc[o.status=='completed','delivery_minutes'].isna().sum())} / {int(o.loc[o.status=='cancelled','delivery_minutes'].isna().sum())}", "0 / 1,199"],
    ["sum of order_amount, completed (INR)", f"{o.loc[o.status=='completed','order_amount'].sum():,.2f}", "2,998,668.80"],
    ["customers: ages missing / min-max age", f"{int(c.age.isna().sum())} / {c.age.min():.0f}-{c.age.max():.0f}", "0 / 18-60"],
    ["customers: gender counts", ", ".join(f"{k} {v}" for k, v in c.gender.value_counts().items()), "female 610, male 560, unknown 30"],
    ["customers: city counts", ", ".join(f"{k} {v}" for k, v in c.city.value_counts().items()), "6 cities"],
], columns=["after-cleaning check", "this run", "reference"])

section = f"""## Part A - Data Foundation

**Raw data audit (task 2).** Shapes: {', '.join(f"{r['table']} {r['rows']}x{r['columns']}" for r in shape_rows)}.

{md_table(audit_df)}

Per-column dtypes and missing values (count and %):

**customers**

{md_table(missing_table(customers))}

**products**

{md_table(missing_table(products))}

**orders**

{md_table(missing_table(orders))}

**tickets** (read with `keep_default_na=False`; blank means empty string)

{md_table(missing_table(tickets))}

**Cleaning rules and how many rows/values each changed (tasks 3-4).**

{md_table(pd.DataFrame({"rule": list(log), "rows / values changed": list(log.values())}))}

{md_table(clean_facts)}

**Storage and validation (task 5).** Clean tables are in `data/clean/*.csv` and in SQLite `data/instamart.db`
(tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text).
`python validate_part_a.py` asserts the reference checkpoints and prints `ALL CHECKPOINTS PASSED`.

**SQL analysis (task 6)** - queries in `sql/queries.sql`, executed with `pd.read_sql_query`.

{md_table(sql_summary)}

Top customer per city (Q6): 

{md_table(top_city)}

Q6 note: RANK() and ROW_NUMBER() differ in exactly {tie_rows} rows because customers who tie on total spend share the same RANK() (and the next rank is skipped) while ROW_NUMBER() always assigns a unique number.

**pandas cross-check (task 7a).** `pd.pivot_table(..., margins=True)` of completed revenue by city x category gives a grand total of INR {grand_total:,.2f}, equal to the SQL Q2 total.

{md_table(pivot, index=True)}

Per-city aggregation (`groupby` + named aggregation):

{md_table(by_city, index=True)}

**EDA (task 7b).** Stats of completed `order_amount`: {', '.join(f"{k} = {v:,.2f}" if not k.startswith('outliers') else f"{k} = {v}" for k, v in stats.items())}.

Correlation matrix (completed orders):

{md_table(corr.round(3), index=True)}

{eda_text}

All amounts in this project are in INR.
"""
write(f"{REP}/part_a.md", section)
print(f"Part A done: {len(o)} clean orders, completed revenue INR {o.loc[o.status=='completed','order_amount'].sum():,.2f}")
