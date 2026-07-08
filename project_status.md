# Customer Churn MLOps — Project Status

Last updated: [Phase 3 complete - extended scope: LR vs XGBoost vs LightGBM, 8 runs, champion selected]

## Completed

### Phase 1 — EDA (notebooks/eda.ipynb)
- Structural checks, target imbalance (73.5% No / 26.5% Yes churn)
- Numeric distributions for tenure, MonthlyCharges, TotalCharges
- Numeric-vs-churn boxplots (lower tenure + higher MonthlyCharges -> more churn)
- TotalCharges data quality fix: 11 rows had blank-space values, all 
  tenure=0 new customers. Fixed via pd.to_numeric(errors='coerce') + 
  fillna(0), not the mean (these customers genuinely have $0 billed so far)
- Full categorical sweep vs churn. Strongest predictors found:
  - Contract type (Month-to-month 42.7% -> One year 11.3% -> Two year 2.8%)
  - InternetService (Fiber optic 41.9% vs DSL 19.0%), confounded with 
    Contract (Fiber customers are 68.7% month-to-month vs DSL's 50.5%)
  - PaymentMethod (Electronic check 45.3% vs 15-19% for others)
  - Add-on services (OnlineSecurity, OnlineBackup, DeviceProtection, 
    TechSupport) - lacking any roughly doubles/triples churn risk
  - Weak predictors: gender, PhoneService, MultipleLines (<3pt gaps)
- Correlation: tenure -0.35, MonthlyCharges +0.19 with churn; 
  tenure/TotalCharges correlated 0.83 with each other (multicollinearity 
  note for linear models, not an issue for tree models)
- IQR outlier check: 0 outliers found on MonthlyCharges/TotalCharges 
  (right-skew widens IQR bounds; not evidence of clean data alone)
- Full findings summary cell at the end of the notebook

### Phase 2 — Preprocessing (src/data_preprocessing.py + configs/config.yaml)
Functions built and verified against real data:
- load_config, load_raw_data, clean_total_charges, drop_identifier_column
- encode_binary_columns (Yes/No + gender -> 0/1)
- encode_ordinal_columns (takes an ordinal_mappings dict - Contract: 
  Month-to-month=0, One year=1, Two year=2)
- encode_nominal_columns (pd.get_dummies, drop_first=True) 
  -- SEE OPEN TODO BELOW
- encode_target_column (Churn Yes/No -> 1/0)
- split_features_target (train_test_split using config's test_size/random_state)
- save_processed_data (writes data/processed/train.csv and test.csv)
- main() chains everything, runnable via `python src/data_preprocessing.py`

config.yaml contains: data paths, target_column, split params 
(test_size: 0.2, random_state: 42), artifacts paths, and feature 
classification (binary_cols, nominal_cols, ordinal_mappings)

Tooling: DVC tracks data/raw/ and data/processed/ (pointer .dvc files 
committed to git, actual data gitignored via consolidated root .gitignore).

### Phase 3 — Model Training + MLflow (src/train.py) — COMPLETE
Functions built and verified against real data (and independently
re-verified against the actual GitHub repo, not just local runs):
- load_processed_data — loads train/test CSVs, splits X/y
- scale_features — StandardScaler fit only on X_train (data leakage
  prevention discussed in depth), transforms both splits, returns
  scaler for reuse in predict.py
- train_logistic_regression, train_xgboost, train_lightgbm — all share
  the identical shape (X_train, y_train, **kwargs) -> fitted model, so
  any of them can be passed interchangeably as train_fn
- evaluate_model — returns accuracy, precision, recall, F1, ROC-AUC
  as a dict
- run_experiment(train_fn, log_fn, params, X_train_scaled, y_train,
  X_test_scaled, y_test) — runs one full MLflow-tracked experiment for
  a given model-training function, its matching MLflow log function
  (mlflow.sklearn.log_model / mlflow.xgboost.log_model /
  mlflow.lightgbm.log_model), and a hyperparameter dict. Both train_fn
  and log_fn are passed in (dependency injection) so run_experiment()
  works across model families without hardcoding any one of them or
  duplicating the mlflow.start_run() block per model type
- main() — top-level orchestrator: loads/scales data once, computes
  scale_pos_weight from y_train (dynamically, not hardcoded, so it
  stays correct if the split ever changes), then calls run_experiment()
  once per configuration being compared. Doesn't call mlflow.* directly
  itself - that's delegated to run_experiment()

**Scope note:** Phase 3 was originally closed out after only two Logistic
Regression runs (baseline vs class_weight='balanced'). This was caught
as an unintentional scope narrowing against the original plan ("compare
models" was meant to include model families, not just LR variants) and
consciously extended (Option A) to properly compare LR vs XGBoost vs
LightGBM before moving to Phase 4.

**8 MLflow runs logged and compared in the `churn_prediction` experiment:**

| Metric | LR Base | LR Bal | XGB Base | XGB Bal | LGBM Base | LGBM Bal | XGB Tuned | LGBM Tuned |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.821 | 0.749 | 0.798 | 0.770 | 0.802 | 0.775 | 0.762 | 0.764 |
| Precision | 0.685 | 0.517 | 0.639 | 0.553 | 0.649 | 0.553 | 0.534 | 0.536 |
| Recall | 0.601 | 0.823 | 0.542 | 0.681 | 0.550 | 0.780 | 0.807 | 0.818 |
| F1 | 0.640 | 0.635 | 0.586 | 0.611 | 0.595 | 0.647 | 0.642 | 0.648 |
| ROC-AUC | 0.862 | 0.862 | 0.842 | 0.838 | 0.849 | 0.851 | 0.856 | 0.857 |

"Base"/"Bal" = default hyperparameters vs. imbalance-corrected
(class_weight='balanced' for LR; scale_pos_weight = negative/positive
class ratio, computed dynamically, for XGBoost/LightGBM). "Tuned" runs
add scale_pos_weight + a quick hand-picked hyperparameter pass
(n_estimators=300, max_depth=4, learning_rate=0.05) - a standard safe
starting combination for tabular data of this size (~7000 rows), not a
full grid search (deferred as a possible future enhancement, not done
in Phase 3).

**Decision: LightGBM Tuned selected as the champion model.**
Params: `{random_state: 42, scale_pos_weight: ~2.766 (dynamic), 
n_estimators: 300, max_depth: 4, learning_rate: 0.05}`
Reasoning: best ROC-AUC (0.857) and best F1 (0.648) of all 8 runs, with
recall (0.818) nearly matching the single highest recall achieved
(LR-balanced, 0.823 - a gap small enough to not be meaningful) while
keeping notably better precision/accuracy than LR-balanced. Represents
the best overall balance across model families rather than an extreme
single-metric optimum.

Runner-up: LR-balanced (highest raw recall, 0.823) - documented as a
close alternative if recall alone were the only priority.

MLflow setup: SQLite backend (mlflow.db) - this is the current MLflow
default for new setups, not a manual config choice. mlruns/ (artifacts)
and mlflow.db (run/metric metadata) both correctly gitignored.

Verification note: local terminal output cross-checked against the
actual GitHub repo (uploaded as a zip) - train.py on GitHub matched
local exactly, config.yaml paths confirmed correct, evaluate.py/
predict.py/tests/test_preprocessing.py/api/app.py confirmed genuinely
empty (correct - reserved for later phases, not a gap).

## Open TODOs (flagged, not yet fixed - relevant for later phases)

**[Phase 5]** encode_nominal_columns uses pd.get_dummies, which derives 
one-hot columns from whatever categories are present in the data given 
to it. Fine for train/test (same source data, no statistical leakage 
since get_dummies doesn't learn from values). NOT fine for predict.py - 
a single new customer record won't have all categories present, so 
get_dummies on it would produce a different/wrong column set than 
training.
Plan: switch to sklearn.preprocessing.OneHotEncoder, fit once on training 
data, save fitted encoder via joblib to config's artifacts.preprocessor_path, 
reuse the same fitted encoder in train.py and predict.py.
(Already documented as a code comment above encode_nominal_columns in the 
actual file too.)

**[Phase 5]** `from data_preprocessing import load_config` (used in both
train.py and data_preprocessing.py) works when running scripts directly
via `python src/train.py`, because Python puts the script's own directory
(src/) on sys.path[0]. This will NOT work automatically the same way once
tests/test_preprocessing.py (a sibling folder to src/) tries to import
from src/ - pytest's import context is different, and either an
src/__init__.py, a conftest.py, or an editable install may be needed.
Confirmed as a real (not yet solved) gotcha to address when Phase 5
testing work begins.

## Not yet started
- **Phase 4 (next): Deep evaluation in src/evaluate.py** — confusion 
  matrix, SHAP values, ROC curve plots, deeper look at the LightGBM 
  Tuned champion model specifically (currently an empty placeholder)
- Phase 5: predict.py, joblib serialization, fix the OneHotEncoder TODO, 
  fix the src/ import path issue, actual tests in 
  tests/test_preprocessing.py
- Phase 6: FastAPI + Docker (api/app.py currently an empty placeholder)
- Phase 7: Evidently monitoring, optional Streamlit dashboard
- Phase 8: README/report polish

## Key learnings & principles (running list)
- **Separation of concerns:** Pure ML functions stay MLflow-agnostic; 
  orchestrator functions (main(), run_experiment()) are the only places 
  that know MLflow/orchestration logic exists.
- **Scaling placement:** Scaling lives in train.py, not 
  data_preprocessing.py, because: CSVs stay human-readable, different 
  models need different treatment (tree-based models are scale-invariant), 
  and scaling is a training-time concern, not a property of the dataset.
- **Threshold vs. ranking quality:** class_weight='balanced' doesn't 
  teach the model new patterns - it changes how mistakes are penalized 
  during training, which shifts the effective decision threshold rather 
  than the model's underlying ability to rank risky customers (why 
  ROC-AUC stayed flat while recall/precision swung significantly).
- **Metric choice should match business cost asymmetry**, not default 
  to F1: missing a churner is costlier than a false alarm here, so 
  recall was weighted more heavily in model selection than F1 parity 
  alone would suggest.
- **Dependency injection for generality:** passing train_fn and log_fn
  as parameters into run_experiment() (rather than hardcoding one
  model/library) is what let one function support Logistic Regression,
  XGBoost, and LightGBM without duplicating the training/logging logic
  per model type.
- **Watch for scope narrowing through momentum:** Phase 3 was initially
  marked "complete" after only comparing LR variants (baseline vs
  balanced), even though the original plan included comparing model
  families. This happened by drifting from one problem (fixing recall)
  straight into closing the phase, without an explicit checkpoint
  asking "does this fully match the original scope?" Worth deliberately
  checking phase scope against the original plan before marking
  something done, not just against what was actually built.
