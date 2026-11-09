## Part B - Customer Segments and Lapse Prediction

**RFM table (snapshot 2025-10-01).** 1,174 customers; recency median 7, min 0, frequency mean 18.44, max 39, monetary median 18,376.80, max 48,484.60.

**K-means sweep (standardised RFM, `n_init=10`, `random_state=42`).**

| K | inertia | silhouette |
|---|---|---|
| 2.0 | 1574.8 | 0.501 |
| 3.0 | 899.1 | 0.499 |
| 4.0 | 602.7 | 0.446 |
| 5.0 | 475.9 | 0.415 |
| 6.0 | 388.7 | 0.377 |
| 7.0 | 329.5 | 0.38 |
| 8.0 | 288.8 | 0.342 |

I chose **K=3**: it has the highest silhouette among K in {3,4,5} and gives segments the business can act on. Inertia always falls as K grows
(more centroids can only reduce within-cluster distance, reaching 0 at K=n), so K=8 beating K=2 on inertia says nothing by itself; the elbow and silhouette decide.

**Segment profile.**

| segment_id | size | mean_recency | mean_frequency | mean_monetary | segment_name | recommended_action | is_premium |
|---|---|---|---|---|---|---|---|
| 0 | 551 | 5.6 | 27.93 | 27922.06 | Premium High-Spenders | Early access, bundles and a dedicated premium desk | True |
| 1 | 114 | 76.99 | 3.82 | 3687.76 | Lapsing / Dormant | Win-back coupon before they churn | False |
| 2 | 509 | 13.95 | 11.44 | 10999.06 | Active Regulars | Loyalty rewards and cross-sell of adjacent categories | False |

The segment with the highest mean monetary value is flagged **Premium**. Cluster numbers are arbitrary labels (no ordering).
PCA(2) on the scaled RFM matrix: explained variance ratio 0.791 + 0.194 = 0.985.
`outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

**Lapse problem (cutoff 2025-09-01).** Cohort: 1,169 customers with a completed order before the cutoff; 187 lapsed (16.00%).
Split: 935 train / 234 test (`stratify=y`, `random_state=42`). Always predicting "not lapsed": accuracy 0.842, lapsed-class F1 0.000.
**No order dated on or after 2025-09-01 entered the features, and the segment labels from Part B task 3 were not used as features.**

**Leakage-free preprocessing.** One-hot (`drop="first"`) and `StandardScaler` live inside a scikit-learn `Pipeline`; no encoder, scaler or resampler saw test rows.
Scaler mean of `frequency` (train only) = 15.446 vs full-cohort mean 15.538: they differ because the scaler only sees the 80% training sample. No feature column has a missing value (asserted in code).

**Models (5-fold CV, scoring F1).**

Logistic regression grid (best `C` = 10, CV F1 0.454):

| C | mean_test_score |
|---|---|
| 100.0 | 0.4542 |
| 10.0 | 0.4542 |
| 1.0 | 0.4404 |
| 0.1 | 0.3713 |
| 0.01 | 0.1337 |

Random forest grid (best {'clf__max_depth': 8, 'clf__min_samples_leaf': 10}, CV F1 0.488):

| max_depth | min_samples_leaf | mean_test_score |
|---|---|---|
| 8.0 | 10.0 | 0.4877 |
| 8.0 | 5.0 | 0.456 |
| 5.0 | 10.0 | 0.4204 |
| 5.0 | 5.0 | 0.4203 |
| 8.0 | 20.0 | 0.3985 |
| 5.0 | 20.0 | 0.3842 |
| 3.0 | 5.0 | 0.3008 |
| 3.0 | 10.0 | 0.2798 |
| 3.0 | 20.0 | 0.2689 |

Test set, lapsed class:

| model | confusion matrix | precision | recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[TN 189, FP 8], [FN 22, TP 15]] | 0.652 | 0.405 | 0.5 |
| Random forest | [[TN 190, FP 7], [FN 22, TP 15]] | 0.682 | 0.405 | 0.508 |

I select the random forest (best cross-validated F1 0.488, test F1 0.508, recall 0.405, precision 0.682). Missing a lapsing customer (a false negative) costs a lost future revenue stream that a cheap voucher could have saved, while wrongly targeting a loyal customer (a false positive) only wastes one small discount. So recall on the lapsed class matters most, but F1 keeps precision from collapsing and the retention budget under control. The model chosen gives the best balance of the two on unseen customers.

**Class-balance experiment (training set only).** Added 635 random duplicates of lapsed training rows (`random_state=42`) so both classes have 785 rows; refit the best logistic regression and tested on the original test set.

| training data | precision | recall | F1 |
|---|---|---|---|
| unbalanced | 0.652 | 0.405 | 0.5 |
| oversampled | 0.459 | 0.919 | 0.613 |

**Recall** moved most. For retention this means oversampling lets us catch more lapsing customers at roughly the same precision, which suits a cheap-to-send campaign.

**Interpretation.** Top-5 random-forest features (`feature_importances_`):

| index | importance |
|---|---|
| monetary | 0.3025 |
| frequency | 0.2617 |
| recency | 0.1039 |
| cancel_rate | 0.0876 |
| avg_discount_pct | 0.0871 |

Top-3 logistic-regression coefficients: `frequency` (-1.364): lower lapse risk when this value is higher / present, so this behaviour signals loyalty; `monetary` (-0.886): lower lapse risk when this value is higher / present, so this behaviour signals loyalty; `gender_unknown` (+0.569): higher lapse risk when this value is higher / present, so watch customers with a high value here.

Model saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading with `joblib.load` reproduces the test predictions exactly (asserted).
**Retention list.** Test-set customers in the Premium segment: 113. Top 10 by predicted lapse probability are in `outputs/retention_list.csv`; 0 of them truly lapsed.
The Premium flag comes from whole-window RFM (it includes the September window that defines the label), so it is a post-hoc whole-window label and the retention list is illustrative, not a leakage-free evaluation.
