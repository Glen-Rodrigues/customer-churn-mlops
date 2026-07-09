# Customer Churn MLOps — Project Status

Last updated: [Phase 4 in progress - MLflow setup fixed, champion model
loading verified, confusion matrix code written (not yet run)]

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
(test_size: 0.2, random_state: 42), artifacts paths, feature 
classification (binary_cols, nominal_cols, ordinal_mappings), and
mlflow settings (see Phase 3/4 notes below).

Tooling: DVC tracks data/raw/ and data/processed/ (pointer .dvc files 
committed to git, actual data gitignored via consolidated root .gitignore).

### Phase 3 — Model Training + MLflow (src/train.py) — COMPLETE
Functions built and verified against real data:
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
  a given model-training function, its matching MLflow log function,
  and a hyperparameter dict. Dependency injection avoids duplicating
  the mlflow.start_run() block per model type.
- main() — top-level orchestrator: loads/scales data once, computes
  scale_pos_weight from y_train dynamically, calls run_experiment()
  once per configuration. Also now explicitly sets the MLflow tracking
  URI from config (see Phase 4 fix below) before set_experiment().

**8 MLflow runs logged and compared in the `churn_prediction` experiment:**

| Metric | LR Base | LR Bal | XGB Base | XGB Bal | LGBM Base | LGBM Bal | XGB Tuned | LGBM Tuned |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.821 | 0.749 | 0.798 | 0.770 | 0.802 | 0.775 | 0.762 | 0.764 |
| Precision | 0.685 | 0.517 | 0.639 | 0.553 | 0.649 | 0.553 | 0.534 | 0.536 |
| Recall | 0.601 | 0.823 | 0.542 | 0.681 | 0.550 | 0.780 | 0.807 | 0.818 |
| F1 | 0.640 | 0.635 | 0.586 | 0.611 | 0.595 | 0.647 | 0.642 | 0.648 |
| ROC-AUC | 0.862 | 0.862 | 0.842 | 0.838 | 0.849 | 0.851 | 0.856 | 0.857 |

**Decision: LightGBM Tuned selected as the champion model.**
Params: `{random_state: 42, scale_pos_weight: ~2.766 (dynamic), 
n_estimators: 300, max_depth: 4, learning_rate: 0.05}`
Reasoning: best F1 (0.648) of all 8 runs, recall (0.818) nearly matching
the single highest recall achieved (LR-balanced, 0.823), while keeping
notably better precision/accuracy than LR-balanced. Note: LightGBM
Tuned's ROC-AUC (0.857) is NOT the single highest across all 8 runs —
both LR variants hit ~0.862, since class_weight='balanced' shifts the
decision threshold rather than the model's underlying ranking ability
(ROC-AUC is threshold-independent). The champion pick was a deliberate
multi-metric judgment call (F1 + recall + precision/accuracy balance),
not a single "max ROC-AUC" rule — this is *why* the MLflow run_id is
stored explicitly in config.yaml rather than auto-selected
programmatically (see Phase 4 notes).

Runner-up: LR-balanced (highest raw recall, 0.823) - documented as a
close alternative if recall alone were the only priority.

## Phase 4 — Deep Evaluation (src/evaluate.py) — IN PROGRESS

### MLflow tracking store cleanup (prerequisite work, now resolved)
- **Root cause found:** `train.py` originally never called
  `mlflow.set_tracking_uri(...)`, so runs silently used MLflow's raw
  file-store default (`mlruns/`). A newer MLflow version refuses to
  serve that file-store backend without an explicit opt-in
  (`MlflowException: filesystem tracking backend is in maintenance
  mode`), which is what first surfaced the issue when evaluate.py
  tried to load the champion model.
- Investigation found `mlflow.db` (SQLite) already existed locally with
  49 runs in `churn_prediction` — turned out to be duplicate copies of
  the same 8 runs (identical ROC-AUC values across many run_ids),
  caused by `train.py` having been re-run multiple times during
  debugging without ever clearing prior runs.
- **Fix applied:** both `mlruns/` and `mlflow.db` were deleted, and
  `train.py`'s `main()` now explicitly calls
  `mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])` right
  after loading config, before `mlflow.set_experiment(...)`. `train.py`
  was re-run once cleanly, producing exactly 8 runs (verified via a
  direct SQLite row-count query).
- **Important clarification (not a bug):** `tracking_uri` only
  controls where run *metadata* (params/metrics/tags) lives. Model
  *artifacts* (the actual serialized model files) still default to a
  local `mlruns/` folder regardless of the tracking backend — the two
  are separate storage concerns and both are needed together. Seeing
  both `mlflow.db` and `mlruns/` populated after a clean run is
  correct, not a sign of duplication.
- `config.yaml` now has an `mlflow:` section (`tracking_uri:
  "sqlite:///mlflow.db"`, `experiment_name: "churn_prediction"`,
  `champion_run_id`) so no MLflow settings are hardcoded in `train.py`
  or `evaluate.py`.

### Champion model identification (post-cleanup)
- New clean champion run: `run_id: f7146820b045405583b8c69498d113ec`,
  run name `polite-moth-290`. (Old run_id
  `43256d3aa45d4c529c132a465cfd1858` from the pre-cleanup duplicated
  store is now stale/invalid — do not reuse.)
- Identified by sorting all 8 runs by ROC-AUC and confirming this run's
  exact metric values match the documented LightGBM Tuned row
  (roc_auc 0.8572036705414721) — notably 3rd highest ROC-AUC out of 8,
  not 1st, consistent with the multi-metric selection reasoning above
  rather than a simple max-ROC-AUC rule.

### evaluate.py — built and verified so far
- `load_champion_model(config)` — loads the champion model directly
  from its MLflow run via `mlflow.lightgbm.load_model()` (using
  `runs:/{run_id}/model`) rather than retraining, so evaluation always
  reflects the exact artifact that was actually selected as champion,
  not a fresh retrained copy that could drift slightly.
- `sanity_check_champion(model, X_test_scaled, y_test)` — re-runs the
  existing `evaluate_model()` (from train.py) on the loaded model and
  prints the result, to confirm the right run was loaded before
  building anything on top of it.
- **Verified: printed metrics from evaluate.py exactly match the
  LightGBM Tuned row** (accuracy 0.7643718949609652, recall
  0.8176943699731903, roc_auc 0.8572036705414721, etc.) — champion
  model loading confirmed correct.
- `get_confusion_matrix(model, X_test_scaled, y_test)` and
  `plot_confusion_matrix(cm, save_path)` — written, added to `main()`,
  saves a labeled heatmap PNG to `artifacts/confusion_matrix.png`.
  **Not yet run/verified by Glen — next immediate step.**

### Design decisions made this phase
- **Load vs retrain the champion model:** chose load-from-MLflow
  (Option B) over retraining fresh in evaluate.py, so evaluation always
  reflects the exact artifact actually compared and selected in
  Phase 3, avoiding drift from re-running training a second time.
- **Champion identification: config-driven run_id vs auto-pick by
  ROC-AUC:** chose storing `champion_run_id` explicitly in
  `config.yaml` over auto-selecting the top ROC-AUC run
  programmatically. Reasoning: the actual champion selection was a
  multi-metric human judgment call (F1 + recall + accuracy balance),
  not a codified single-metric rule — auto-picking "max ROC-AUC" would
  not have even selected the actual champion (it would have picked one
  of the LR runs instead, confirmed when the new run's ROC-AUC ranking
  was checked). Revisit auto-selection only if/when Phase 7 introduces
  automated retraining with a codified selection rule.

## Open TODOs (flagged, not yet fixed - relevant for later phases)

**[Phase 5]** encode_nominal_columns uses pd.get_dummies, which derives 
one-hot columns from whatever categories are present in the data given 
to it. Fine for train/test (same source data, no statistical leakage). 
NOT fine for predict.py - a single new customer record won't have all 
categories present, so get_dummies on it would produce a different/wrong 
column set than training.
Plan: switch to sklearn.preprocessing.OneHotEncoder, fit once on training 
data, save fitted encoder via joblib to config's artifacts.preprocessor_path, 
reuse the same fitted encoder in train.py and predict.py.

**[Phase 5]** `scale_features()` in train.py fits a StandardScaler on
X_train and returns it (per its own docstring, "for reuse in predict.py"),
but train.py's main() never actually saves that fitted scaler anywhere.
Harmless today - evaluate.py re-fits an identical scaler on the same
X_train, which is mathematically safe since StandardScaler has no
randomness and the data doesn't change. NOT safe for predict.py: a single
new customer record has no "training data" to fit a fresh scaler on, so
the exact fitted scaler must be persisted and reloaded, not recomputed.
Plan: save via joblib to a config path (same pattern as the OneHotEncoder
TODO above - possibly the same artifacts.preprocessor_path, or a
dedicated scaler_path), fit once in train.py, reuse in predict.py.

**[Phase 5]** `from data_preprocessing import load_config` (used in both
train.py and data_preprocessing.py) works when running scripts directly
via `python src/train.py`, but will NOT work automatically the same way
once tests/test_preprocessing.py (a sibling folder to src/) tries to
import from src/ - pytest's import context is different. Confirmed as a
real (not yet solved) gotcha to address when Phase 5 testing work begins.

**[Minor, cosmetic]** `evaluate.py` prints a sklearn UserWarning
("X does not have valid feature names, but LGBMClassifier was fitted
with feature names") because `scale_features()`'s StandardScaler
strips column names from the DataFrame, returning a plain NumPy array.
Harmless — column order is preserved so predictions are still correct.
Optional fix later: `StandardScaler.set_output(transform="pandas")`.

## Not yet started (remainder of Phase 4)
- Run and verify `get_confusion_matrix` / `plot_confusion_matrix`
  against real data, interpret the FN ("missed churners") count
- ROC curve plot for the champion model
- SHAP values for feature importance/explainability, specific to the
  LightGBM Tuned champion

## Not yet started (later phases)
- Phase 5: predict.py, joblib serialization, fix the OneHotEncoder TODO, 
  fix the src/ import path issue, actual tests in 
  tests/test_preprocessing.py
- Phase 6: FastAPI + Docker (api/app.py currently an empty placeholder)
- Phase 7: Evidently monitoring, optional Streamlit dashboard. If automated
  retraining + auto-selection of the champion (by e.g. highest ROC-AUC) is
  introduced here, note that `load_champion_model()` in evaluate.py
  currently hardcodes `mlflow.lightgbm.load_model()`, which only works
  because the champion is manually fixed to a known LightGBM run
  (config-driven `champion_run_id`). An auto-picked champion could be any
  of the 3 model families (confirmed: a pure max-ROC-AUC rule would
  actually pick Logistic Regression here, not LightGBM - see Phase 3/4
  notes) - LightGBM's loader can't load an XGBoost or sklearn artifact, so
  the loader would need to either (a) switch to the generic
  `mlflow.pyfunc.load_model()` (works for any flavor, but loses native
  model access - e.g. feature_importances_ - needed for SHAP), or
  (b) dynamically dispatch to the correct flavor-specific loader based on
  the run's logged metadata, mirroring the train_fn/log_fn dependency
  injection pattern already used in run_experiment().
- Phase 8: README/report polish

## Key learnings & principles (running list)
- **Separation of concerns:** Pure ML functions stay MLflow-agnostic; 
  orchestrator functions (main(), run_experiment()) are the only places 
  that know MLflow/orchestration logic exists.
- **Scaling placement:** Scaling lives in train.py, not 
  data_preprocessing.py — CSVs stay human-readable, tree-based models 
  are scale-invariant, and scaling is a training-time concern, not a 
  property of the dataset.
- **Threshold vs. ranking quality:** class_weight='balanced' doesn't 
  teach the model new patterns - it changes how mistakes are penalized 
  during training, shifting the effective decision threshold rather 
  than the model's underlying ability to rank risky customers (why 
  ROC-AUC stayed flat between LR baseline/balanced while recall/precision 
  swung significantly — same pattern later confirmed with LightGBM
  Tuned's ROC-AUC not being the single highest across all 8 runs).
- **Metric choice should match business cost asymmetry**, not default 
  to F1: missing a churner is costlier than a false alarm here, so 
  recall was weighted more heavily in model selection than F1 parity 
  alone would suggest.
- **Dependency injection for generality:** passing train_fn and log_fn
  as parameters into run_experiment() let one function support LR,
  XGBoost, and LightGBM without duplicating training/logging logic.
- **Watch for scope narrowing through momentum:** caught once in Phase 3
  (initially closing the phase after only comparing LR variants).
  Worth deliberately checking phase scope against the original plan
  before marking something done.
- **Load persisted artifacts, don't silently re-derive them:** loading
  the champion model from MLflow (Phase 4) rather than retraining is
  the same underlying principle as saving the fitted scaler/encoder for
  reuse in predict.py (Phase 5 TODO) — anything that was fit/selected
  once should be persisted and reloaded, not silently recomputed
  downstream where subtle drift could creep in unnoticed.
- **Tracking-URI ≠ artifact storage in MLflow:** setting
  `mlflow.set_tracking_uri()` only controls metadata storage location;
  model artifacts still default to a local `mlruns/` folder unless
  separately configured. Seeing both a metadata store and an `mlruns/`
  folder populated is expected, not duplication.
- **Config-driven vs auto-selected "champion":** worth deciding
  deliberately whether a downstream script should auto-derive a value
  (e.g. "best ROC-AUC run") or read an explicitly stored decision — the
  right choice depends on whether the original selection was itself a
  codified single-metric rule or a human multi-metric judgment call.
