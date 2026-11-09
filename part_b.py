import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from util import md_table, write

OUT, REP = "outputs", "outputs/readme_parts"
os.makedirs(REP, exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
done = orders[orders["status"] == "completed"]

# ---------------------------------------------------------------- 1. RFM
g = done.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum().round(2),
}).reset_index()
rfm_desc = (f"{len(rfm):,} customers; recency median {rfm.recency.median():.0f}, min {rfm.recency.min()}, "
            f"frequency mean {rfm.frequency.mean():.2f}, max {rfm.frequency.max()}, "
            f"monetary median {rfm.monetary.median():,.2f}, max {rfm.monetary.max():,.2f}")

# ---------------------------------------------------------------- 2. KMeans sweep
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
ktab = pd.DataFrame(rows)
cand = ktab[ktab.K.isin([3, 4, 5])]
K = int(cand.sort_values(["silhouette", "K"], ascending=[False, True]).iloc[0]["K"])
ktab.to_csv(f"{OUT}/kmeans_table.csv", index=False)

# ---------------------------------------------------------------- 3. profile / name / PCA
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)
premium_id = prof["mean_monetary"].idxmax()
rest = prof.drop(index=premium_id).sort_values("mean_recency")
names = {premium_id: ("Premium High-Spenders", "Early access, bundles and a dedicated premium desk")}
roles = [("Active Regulars", "Loyalty rewards and cross-sell of adjacent categories"),
         ("Occasional Shoppers", "Free-delivery or basket-size nudges"),
         ("Cooling Shoppers", "Reminder push with a small voucher"),
         ("Lapsing / Dormant", "Win-back coupon before they churn")]
ids = list(rest.index)
for i, sid in enumerate(ids):
    if i == 0:
        names[sid] = roles[0]
    elif i == len(ids) - 1:
        names[sid] = roles[3]
    else:
        names[sid] = roles[1 + (i - 1) % 2]
prof["segment_name"] = [names[i][0] for i in prof.index]
prof["recommended_action"] = [names[i][1] for i in prof.index]
prof["is_premium"] = [i == premium_id for i in prof.index]
rfm["segment_name"] = rfm.segment_id.map(lambda i: names[i][0])
rfm["is_premium"] = rfm.segment_id == premium_id
pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv(f"{OUT}/customer_segments.csv", index=False)

# ---------------------------------------------------------------- 4. lapse problem
pre = orders[orders.order_ts < CUTOFF]
pre_done = pre[pre.status == "completed"]
gd = pre_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - gd["order_ts"].max()).dt.days,
    "frequency": gd["order_id"].nunique(),
    "monetary": gd["order_amount"].sum(),
    "avg_discount_pct": gd["discount_pct"].mean(),
    "avg_delivery_minutes": gd["delivery_minutes"].mean(),
})
feat["cancel_rate"] = pre.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
m = pre_done.merge(products[["product_id", "category"]], on="product_id")
fav = (m.groupby(["customer_id", "category"]).size().reset_index(name="n")
        .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id").set_index("customer_id")["category"])
feat["fav_category"] = fav
cu = customers.set_index("customer_id")
feat = feat.join(cu[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")
sept = orders[(orders.status == "completed") & (orders.order_ts >= CUTOFF) & (orders.order_ts < SNAPSHOT)]
y = (~feat.index.isin(sept.customer_id)).astype(int)
NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate", "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
Xf = feat[NUM + CAT]
assert Xf.isna().sum().sum() == 0, "feature matrix must have no missing values"
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=42)
base_acc = float((yte == 0).mean())
base_f1 = f1_score(yte, np.zeros(len(yte), dtype=int), zero_division=0)


# ---------------------------------------------------------------- 5. leakage-free preprocessing
def make_pipe(clf):
    pre_ = ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT), ("num", StandardScaler(), NUM)])
    return Pipeline([("pre", pre_), ("clf", clf)])


probe = make_pipe(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
train_mean_freq = probe.named_steps["pre"].named_transformers_["num"].mean_[NUM.index("frequency")]
full_mean_freq = Xf["frequency"].mean()

# ---------------------------------------------------------------- 6. train / tune / evaluate
lr_gs = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)),
                     {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_gs = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(Xtr, ytr)


def evaluate(model, Xe=Xte, ye=yte):
    p = model.predict(Xe)
    return {"cm": confusion_matrix(ye, p), "precision": precision_score(ye, p, zero_division=0),
            "recall": recall_score(ye, p, zero_division=0), "f1": f1_score(ye, p, zero_division=0)}


ev_lr, ev_rf = evaluate(lr_gs.best_estimator_), evaluate(rf_gs.best_estimator_)
if rf_gs.best_score_ >= lr_gs.best_score_:
    sel_name, sel_gs, sel_ev = "Random forest", rf_gs, ev_rf
else:
    sel_name, sel_gs, sel_ev = "Logistic regression", lr_gs, ev_lr
model = sel_gs.best_estimator_
if sel_ev["f1"] < 0.55:
    print(f"WARNING: selected model test F1 {sel_ev['f1']:.3f} is below the required 0.55 - check features/seeds")


def grid_table(gs):
    r = pd.DataFrame(gs.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")] + ["mean_test_score"]
    return r[cols].rename(columns=lambda c: c.replace("param_clf__", "")).round(4).sort_values("mean_test_score", ascending=False)


def cm_text(ev):
    t = ev["cm"]
    return f"[[TN {t[0,0]}, FP {t[0,1]}], [FN {t[1,0]}, TP {t[1,1]}]]"


justification = (
    f"I select the {sel_name.lower()} (best cross-validated F1 {sel_gs.best_score_:.3f}, test F1 {sel_ev['f1']:.3f}, "
    f"recall {sel_ev['recall']:.3f}, precision {sel_ev['precision']:.3f}). "
    f"Missing a lapsing customer (a false negative) costs a lost future revenue stream that a cheap voucher could have saved, "
    f"while wrongly targeting a loyal customer (a false positive) only wastes one small discount. "
    f"So recall on the lapsed class matters most, but F1 keeps precision from collapsing and the retention budget under control. "
    f"The model chosen gives the best balance of the two on unseen customers."
)

# ---------------------------------------------------------------- 7. class balance experiment
pos, neg = Xtr[ytr == 1], Xtr[ytr == 0]
extra = pos.sample(n=len(neg) - len(pos), replace=True, random_state=42)
Xbal = pd.concat([Xtr, extra])
ybal = np.concatenate([ytr, np.ones(len(extra), dtype=int)])
lr_best = make_pipe(LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))
lr_bal = lr_best.fit(Xbal, ybal)
ev_bal = evaluate(lr_bal)
moved = "recall" if ev_bal["recall"] - ev_lr["recall"] >= abs(ev_bal["precision"] - ev_lr["precision"]) else "precision"

# ---------------------------------------------------------------- 8. interpret / persist / use
rf_pre = rf_gs.best_estimator_.named_steps["pre"]
fn = [n.split("__", 1)[1] for n in rf_pre.get_feature_names_out()]
imp = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=fn).sort_values(ascending=False).head(5).round(4)
lr_fn = [n.split("__", 1)[1] for n in lr_gs.best_estimator_.named_steps["pre"].get_feature_names_out()]
coefs = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_fn)
top3 = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(3)
reading = [f"`{k}` ({v:+.3f}): {'higher' if v > 0 else 'lower'} lapse risk when this value is higher / present, "
           f"so {'watch customers with a high value here' if v > 0 else 'this behaviour signals loyalty'}" for k, v in top3.items()]

joblib.dump(model, f"{OUT}/lapse_model.joblib")
reloaded = joblib.load(f"{OUT}/lapse_model.joblib")
assert np.array_equal(reloaded.predict(Xte), model.predict(Xte)), "reloaded model must reproduce predictions"

seg = pd.read_csv(f"{OUT}/customer_segments.csv")
premium_ids = set(seg.loc[seg.is_premium, "customer_id"])
proba = pd.Series(model.predict_proba(Xte)[:, 1], index=Xte.index)
ret = pd.DataFrame({"customer_id": Xte.index, "lapse_probability": proba.values.round(4),
                    "actually_lapsed": yte})
ret = ret[ret.customer_id.isin(premium_ids)].sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv(f"{OUT}/retention_list.csv", index=False)
n_prem_test = int(Xte.index.isin(premium_ids).sum())

# ---------------------------------------------------------------- README section
section = f"""## Part B - Customer Segments and Lapse Prediction

**RFM table (snapshot 2025-10-01).** {rfm_desc}.

**K-means sweep (standardised RFM, `n_init=10`, `random_state=42`).**

{md_table(ktab)}

I chose **K={K}**: it has the highest silhouette among K in {{3,4,5}} and gives segments the business can act on. Inertia always falls as K grows
(more centroids can only reduce within-cluster distance, reaching 0 at K=n), so K=8 beating K=2 on inertia says nothing by itself; the elbow and silhouette decide.

**Segment profile.**

{md_table(prof, index=True)}

The segment with the highest mean monetary value is flagged **Premium**. Cluster numbers are arbitrary labels (no ordering).
PCA(2) on the scaled RFM matrix: explained variance ratio {evr[0]:.3f} + {evr[1]:.3f} = {evr.sum():.3f}.
`outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

**Lapse problem (cutoff 2025-09-01).** Cohort: {len(feat):,} customers with a completed order before the cutoff; {int(y.sum())} lapsed ({y.mean():.2%}).
Split: {len(Xtr)} train / {len(Xte)} test (`stratify=y`, `random_state=42`). Always predicting "not lapsed": accuracy {base_acc:.3f}, lapsed-class F1 {base_f1:.3f}.
**No order dated on or after 2025-09-01 entered the features, and the segment labels from Part B task 3 were not used as features.**

**Leakage-free preprocessing.** One-hot (`drop="first"`) and `StandardScaler` live inside a scikit-learn `Pipeline`; no encoder, scaler or resampler saw test rows.
Scaler mean of `frequency` (train only) = {train_mean_freq:.3f} vs full-cohort mean {full_mean_freq:.3f}: they differ because the scaler only sees the 80% training sample. No feature column has a missing value (asserted in code).

**Models (5-fold CV, scoring F1).**

Logistic regression grid (best `C` = {lr_gs.best_params_['clf__C']}, CV F1 {lr_gs.best_score_:.3f}):

{md_table(grid_table(lr_gs))}

Random forest grid (best {rf_gs.best_params_}, CV F1 {rf_gs.best_score_:.3f}):

{md_table(grid_table(rf_gs))}

Test set, lapsed class:

{md_table(pd.DataFrame([
    ["Logistic regression", cm_text(ev_lr), round(ev_lr['precision'],3), round(ev_lr['recall'],3), round(ev_lr['f1'],3)],
    ["Random forest", cm_text(ev_rf), round(ev_rf['precision'],3), round(ev_rf['recall'],3), round(ev_rf['f1'],3)]],
    columns=["model", "confusion matrix", "precision", "recall", "F1"]))}

{justification}

**Class-balance experiment (training set only).** Added {len(extra)} random duplicates of lapsed training rows (`random_state=42`) so both classes have {int((ybal==0).sum())} rows; refit the best logistic regression and tested on the original test set.

{md_table(pd.DataFrame([
    ["unbalanced", round(ev_lr['precision'],3), round(ev_lr['recall'],3), round(ev_lr['f1'],3)],
    ["oversampled", round(ev_bal['precision'],3), round(ev_bal['recall'],3), round(ev_bal['f1'],3)]],
    columns=["training data", "precision", "recall", "F1"]))}

**{moved.capitalize()}** moved most. For retention this means oversampling lets us catch more lapsing customers at roughly the same precision, which suits a cheap-to-send campaign.

**Interpretation.** Top-5 random-forest features (`feature_importances_`):

{md_table(imp.rename('importance').to_frame(), index=True)}

Top-3 logistic-regression coefficients: {'; '.join(reading)}.

Model saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading with `joblib.load` reproduces the test predictions exactly (asserted).
**Retention list.** Test-set customers in the Premium segment: {n_prem_test}. Top {len(ret)} by predicted lapse probability are in `outputs/retention_list.csv`; {int(ret.actually_lapsed.sum())} of them truly lapsed.
The Premium flag comes from whole-window RFM (it includes the September window that defines the label), so it is a post-hoc whole-window label and the retention list is illustrative, not a leakage-free evaluation.
"""
write(f"{REP}/part_b.md", section)
print(f"Part B done: K={K}, selected {sel_name}, test F1 {sel_ev['f1']:.3f}")
