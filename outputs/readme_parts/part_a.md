## Part A - Data Foundation

**Raw data audit (task 2).** Shapes: customers 1200x5, products 40x4, orders 25576x9, tickets 164x5.

| audit figure | value |
|---|---|
| orders raw rows | 25576 |
| exact duplicate rows | 227 |
| rows with status 'pending' | 1227 |
| distinct order_id | 24122 |
| order_amount not convertible to a number | 224 |
| negative delivery_minutes | 0 |
| blank delivery_minutes | 8078 |
| customers: blank age | 65 |
| customers: age outside 10-100 | 385 |
| customers: distinct raw city spellings | 18 |
| customers: distinct non-blank raw gender spellings | 4 |
| customers: blank gender | 222 |

Per-column dtypes and missing values (count and %):

**customers**

| column | dtype | missing | missing_pct |
|---|---|---|---|
| customer_id | str | 0 | 0.0 |
| city | str | 0 | 0.0 |
| signup_date | str | 0 | 0.0 |
| age | float64 | 65 | 5.42 |
| gender | str | 222 | 18.5 |

**products**

| column | dtype | missing | missing_pct |
|---|---|---|---|
| product_id | str | 0 | 0.0 |
| product_name | str | 0 | 0.0 |
| category | str | 0 | 0.0 |
| price_inr | int64 | 0 | 0.0 |

**orders**

| column | dtype | missing | missing_pct |
|---|---|---|---|
| order_id | str | 0 | 0.0 |
| customer_id | str | 0 | 0.0 |
| product_id | str | 0 | 0.0 |
| quantity | int64 | 0 | 0.0 |
| order_ts | str | 0 | 0.0 |
| status | str | 0 | 0.0 |
| discount_pct | int64 | 0 | 0.0 |
| order_amount | str | 110 | 0.43 |
| delivery_minutes | float64 | 8078 | 31.58 |

**tickets** (read with `keep_default_na=False`; blank means empty string)

| column | dtype | missing | missing_pct |
|---|---|---|---|
| ticket_id | str | 0 | 0.0 |
| customer_id | str | 0 | 0.0 |
| order_id | str | 164 | 100.0 |
| text | str | 4 | 2.44 |
| true_category | str | 0 | 0.0 |

**Cleaning rules and how many rows/values each changed (tasks 3-4).**

| rule | rows / values changed |
|---|---|
| customers: city spellings -> clean cities | 18 -> 6 |
| customers: gender values mapped to male/female/unknown | 4 spellings + 222 blanks -> 3 values |
| customers: blank ages imputed with city median | 65 |
| customers: ages outside 10-100 set to missing, then imputed | 385 |
| orders: exact duplicate rows removed | 227 |
| orders: superseded (earlier 'pending') records removed, latest order_ts kept | 1227 |
| orders: order_amount unparseable -> recomputed from price x qty x (1-disc) | 216 |
| orders: negative delivery_minutes set to missing | 0 |
| orders: completed orders' missing delivery_minutes -> category median (completed) | 4307 |
| orders: cancelled orders' delivery_minutes left missing | 2477 |

| after-cleaning check | this run | reference |
|---|---|---|
| orders rows (one per order_id) | 24122 | 12,309 |
| completed / cancelled | 21,645 / 2,477 | 11,110 / 1,199 |
| pending rows | 0 | 0 |
| missing order_amount | 0 | 0 |
| missing delivery_minutes (completed / cancelled) | 0 / 2477 | 0 / 1,199 |
| sum of order_amount, completed (INR) | 21,403,983.10 | 2,998,668.80 |
| customers: ages missing / min-max age | 0 / 25-50 | 0 / 18-60 |
| customers: gender counts | female 493, male 485, unknown 222 | female 610, male 560, unknown 30 |
| customers: city counts | Pune 218, Chennai 207, Hyderabad 202, Delhi 201, Bengaluru 191, Mumbai 181 | 6 cities |

**Storage and validation (task 5).** Clean tables are in `data/clean/*.csv` and in SQLite `data/instamart.db`
(tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text).
`python validate_part_a.py` asserts the reference checkpoints and prints `ALL CHECKPOINTS PASSED`.

**SQL analysis (task 6)** - queries in `sql/queries.sql`, executed with `pd.read_sql_query`.

| query | rows returned | reference / key result |
|---|---|---|
| Q1 WHERE then HAVING | 885 | 30 customers |
| Q2 three-table join | 30 | 30 rows; largest = Pune / Grocery INR 928,992.00 (ref. Bengaluru / Electronics 303,909.45) |
| Q3 scalar subquery | 21 | 11 products |
| Q4 LEFT JOIN | 26 | 25 customers |
| Q5 CTE | 592 | 550 customers; average INR 18,231.67 (ref. 2,552.06) |
| Q6 window functions | 1174 | 2 rows where RANK() differs from ROW_NUMBER() (ref. 8) |
| Q7 running total | 183 | 183 daily rows; last cumulative INR 21,403,983.10 (ref. 2,998,668.80) |

Top customer per city (Q6): 

| city | customer_id | total_spend |
|---|---|---|
| Bengaluru | C1184 | 48484.6 |
| Chennai | C0659 | 44902.2 |
| Delhi | C0468 | 41745.0 |
| Hyderabad | C0967 | 46512.0 |
| Mumbai | C0387 | 43364.6 |
| Pune | C1135 | 45430.8 |

Q6 note: RANK() and ROW_NUMBER() differ in exactly 2 rows because customers who tie on total spend share the same RANK() (and the next rank is skipped) while ROW_NUMBER() always assigns a unique number.

**pandas cross-check (task 7a).** `pd.pivot_table(..., margins=True)` of completed revenue by city x category gives a grand total of INR 21,403,983.10, equal to the SQL Q2 total.

| city | Dairy | Electronics | Grocery | Personal Care | Snacks | Total |
|---|---|---|---|---|---|---|
| Bengaluru | 736861.7 | 701236.8 | 667399.4 | 779261.2 | 538773.4 | 3423532.5 |
| Chennai | 637076.4 | 702788.4 | 764723.9 | 788845.9 | 632044.7 | 3525479.3 |
| Delhi | 650726.4 | 712134.7 | 737359.0 | 830296.1 | 604846.2 | 3535362.4 |
| Hyderabad | 700219.2 | 702721.9 | 745315.4 | 861818.0 | 612780.6 | 3622855.1 |
| Mumbai | 650239.1 | 633581.3 | 689621.1 | 740653.2 | 587878.0 | 3301972.7 |
| Pune | 688261.1 | 786644.5 | 928992.0 | 920875.1 | 670008.4 | 3994781.1 |
| Total | 4063383.9 | 4239107.6 | 4533410.8 | 4921749.5 | 3646331.3 | 21403983.1 |

Per-city aggregation (`groupby` + named aggregation):

| city | order_sum | order_mean | order_count |
|---|---|---|---|
| Bengaluru | 3423532.5 | 1006.92 | 3400 |
| Chennai | 3525479.3 | 994.21 | 3546 |
| Delhi | 3535362.4 | 983.14 | 3596 |
| Hyderabad | 3622855.1 | 974.67 | 3717 |
| Mumbai | 3301972.7 | 979.81 | 3370 |
| Pune | 3994781.1 | 994.72 | 4016 |

**EDA (task 7b).** Stats of completed `order_amount`: mean = 988.87, median = 898.00, skewness = 0.66, Q1 = 359.20, Q3 = 1,465.60, lower fence = -1,300.40, upper fence = 3,125.20, outliers above = 0, outliers below = 0.

Correlation matrix (completed orders):

| index | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.0 | -0.109 | 0.439 | -0.01 |
| discount_pct | -0.109 | 1.0 | 0.006 | -0.004 |
| quantity | 0.439 | 0.006 | 1.0 | 0.006 |
| delivery_minutes | -0.01 | -0.004 | 0.006 | 1.0 |

**What is this data?** 21,645 completed Instamart orders from 01 Apr 2025 to 30 Sep 2025, with amount (INR), quantity, discount and delivery time. **Distribution:** order_amount is right-skewed (mean 988.87 vs median 898.00, skewness 0.66); half of orders lie between 359.20 and 1465.60 at the quartiles. **Outliers:** 0 orders exceed the upper 1.5xIQR fence of 3125.20 and 0 fall below the lower fence; they look like genuine high-priced baskets (likely electronics), not errors. **Relationships:** the largest absolute correlation is order_amount with quantity (r = +0.439); r of this size is weak and does not establish causation.

All amounts in this project are in INR.
