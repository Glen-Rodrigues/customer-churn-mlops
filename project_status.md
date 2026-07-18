# Customer Churn MLOps — Project Status

Last updated: [Phase 6 in progress - api/app.py built and verified
end-to-end (unchanged from before). Dockerfile now WRITTEN and fully
commented, but NOT yet built or run - blocked on Docker Desktop
installation on the local Windows machine (also runs VMware
Workstation; confirmed modern VMware + WSL2/Hyper-V coexistence is
supported, proceeding with install). `docker build` has not been
executed yet - that's the next step once Docker install is verified
working.]

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

## Phase 4 — Deep Evaluation (src/evaluate.py) — COMPLETE

### MLflow tracking store cleanup (prerequisite work, resolved)
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
  was re-run once cleanly, producing exactly 8 runs.
- **Clarification (not a bug):** `tracking_uri` only controls where run
  *metadata* (params/metrics/tags) lives. Model *artifacts* still
  default to a local `mlruns/` folder regardless of tracking backend —
  seeing both populated after a clean run is correct, not duplication.
- `config.yaml` now has an `mlflow:` section (`tracking_uri:
  "sqlite:///mlflow.db"`, `experiment_name: "churn_prediction"`,
  `champion_run_id`) so no MLflow settings are hardcoded.

### Champion model identification
- Clean champion run: `run_id: f7146820b045405583b8c69498d113ec`, run
  name `polite-moth-290`. (Old run_id `43256d3aa45d4c529c132a465cfd1858`
  from the pre-cleanup duplicated store is stale/invalid.)
- **[Updated in Phase 5]** MLflow store was deleted and train.py re-run
  after the OneHotEncoder rework (see Phase 5 below), producing a new
  run_id: `d7d6aea0f5bc4289a37aad468b751428`. Metrics verified
  identical to the original champion (roc_auc 0.8572, f1 0.6476) before
  updating config.yaml. This is now the current `champion_run_id`.
- Confirmed by matching this run's metrics exactly to the documented
  LightGBM Tuned row (roc_auc 0.8572036705414721) — 3rd highest ROC-AUC
  out of 8, not 1st, consistent with the multi-metric selection
  reasoning (see Phase 3), not a max-ROC-AUC rule.

### evaluate.py — built, run, and verified end-to-end against real data
All functions verified with actual output, restructured into a `main()`
orchestrator matching train.py's pattern (functions do the work, main()
chains them, `if __name__ == "__main__": main()`).

- `load_champion_model(config)` — loads via `mlflow.lightgbm.load_model()`
  (`runs:/{run_id}/model`). **Verified:** printed model params exactly
  match config (`n_estimators=300, max_depth=4, learning_rate=0.05,
  scale_pos_weight≈2.766`).
- `sanity_check_champion(model, X_test_scaled, y_test)` — reruns
  train.py's `evaluate_model()`. **Verified:** metrics exactly match
  Phase 3's LightGBM Tuned row (accuracy 0.7644, precision 0.5360,
  recall 0.8177, f1 0.6476, roc_auc 0.8572).
- `get_confusion_matrix()` / `plot_confusion_matrix()` — saved to
  `artifacts/confusion_matrix.png`. **Verified result:**
  `[[772 FP:264], [FN:68 TP:305]]`. Cross-checked: recall
  (305/373=0.8177) and precision (305/569=0.536) both match sanity
  check exactly.
- `plot_roc_curve()` — saved to `artifacts/roc_curve.png`, AUC computed
  directly from the curve via `auc(fpr, tpr)` so the plotted line and
  printed value are guaranteed consistent. Uses `predict_proba`, not
  `predict`, since the point is pre-threshold performance.
- `compute_shap_values()` / `plot_shap_summary()` — `shap.TreeExplainer`
  (exact + fast for tree models, vs. slower/approximate general
  `Explainer`). **Fix applied:** newer SHAP versions return a list of
  two arrays (`[class_0, class_1]`) for binary classifiers instead of
  one array; added `isinstance(shap_values, list)` check to select
  index `[1]` (Churn class), matching `encode_target_column()`'s
  Yes→1 mapping. Saved to `artifacts/shap_summary.png`.
  **Verified against Phase 1 EDA:** top SHAP features (Contract,
  tenure, MonthlyCharges, InternetService_Fiber optic,
  OnlineSecurity_Yes, TechSupport_Yes) and their directionality all
  match EDA's independently-found churn predictors and directions —
  the model learned the same signal the manual EDA surfaced. Weak-
  predictor features from EDA (gender, Partner, Dependents,
  PhoneService) also show near-zero SHAP spread, consistent with EDA.
- `analyze_false_negatives()` — compares the 68 missed churners (FN)
  against the 305 correctly-caught churners (TP) on unscaled `X_test`
  (real units, not standardized scores). **Finding:** missed churners
  look "safe" by the model's own logic — ~2.3x higher TotalCharges
  (2662 vs 1145), ~2.5x longer tenure (32.5 vs 12.9 months), far more
  likely to be on longer contracts (avg Contract 0.559 vs 0.016) and to
  already have OnlineSecurity/TechSupport. This is a genuine model
  blind spot, not a bug: the dataset has no signal for why an otherwise
  "safe-profile" customer churns (price sensitivity, competitor offers,
  support experience aren't captured in this data).
- `compare_feature_importance()` — cross-checks SHAP's mean |impact|
  ranking against LightGBM's built-in `feature_importances_` (split-
  count based). **Finding:** top 3 features (Contract, tenure,
  MonthlyCharges/TotalCharges) agree between both methods, but ranking
  *order* differs — LightGBM ranks MonthlyCharges #1 by split-count
  (1002 splits, used often for small refinements) while SHAP ranks
  Contract #1 by actual prediction impact (fewer splits, 193, but each
  one swings the prediction more). Frequency of use ≠ size of impact.

### Design decisions made this phase
- **Load vs retrain the champion model:** chose load-from-MLflow over
  retraining fresh, so evaluation reflects the exact artifact actually
  selected in Phase 3, avoiding drift.
- **Config-driven run_id vs auto-pick by ROC-AUC:** chose storing
  `champion_run_id` explicitly over auto-selecting top ROC-AUC.
  Reasoning: champion selection was a multi-metric human judgment call
  (F1 + recall + accuracy balance) — auto-picking "max ROC-AUC" would
  not have selected the actual champion (it would have picked an LR
  run instead, both LR variants hit ROC-AUC ~0.862 vs LightGBM Tuned's
  0.857). Revisit only if/when Phase 7 introduces automated retraining
  with a codified selection rule.
- **Re-fitting StandardScaler in evaluate.py (not loading a persisted
  one):** confirmed safe here — no randomness, identical X_train data,
  so the re-fit scaler is mathematically identical to train.py's.
  Explicitly NOT the same situation as Phase 5's predict.py, where a
  single new customer record has no training data to re-fit on (see
  Open TODOs).

## Phase 5 — predict.py Infrastructure — COMPLETE

### data_preprocessing.py / train.py restructuring
- Nominal encoding moved out of data_preprocessing.py entirely - it now
  stops after binary/ordinal encoding + target encoding, leaving nominal
  columns as raw strings all the way through train.csv/test.csv.
- train.py's main() now: loads data -> encode_nominal_features (fit on
  X_train only) -> scale_features -> persist encoder + scaler via joblib
  -> proceed with the existing 8-run MLflow comparison, unchanged.
- Full re-run verified byte-for-byte equivalent metrics across all 8
  runs vs. the pre-refactor table (see Phase 3), confirming the
  OneHotEncoder swap didn't change model behavior, only fixed the
  predict.py-breaking leakage/robustness issue.
- config.yaml's champion_run_id updated to the new post-refactor run_id
  (see Phase 4 section above for the specific value and verification).

### evaluate.py updates (kept in sync with the pipeline change)
- Added load_preprocessing_artifacts() (joblib.load for encoder +
  scaler) and apply_preprocessing() (transform-only, no fitting) -
  written explicitly for reuse in predict.py, not just for evaluate.py's
  own needs.
- main() updated to call these instead of re-deriving encoded/scaled
  data inline; all downstream calls (SHAP, false-negative analysis,
  importance comparison) updated to use the now-larger encoded column
  set (X_train_encoded.columns, 29 cols after one-hot expansion) instead
  of the original 19 raw columns. Verified: identical output to the
  pre-refactor run across every printed number and saved plot.

### predict.py — built and verified against real data
- `load_artifacts(config)` - loads champion model, encoder, scaler, AND
  the canonical raw feature column order (X_train.columns, loaded once
  from train.csv). Returns all four.
- `preprocess_customer_data(df, config, encoder, scaler, feature_columns)`
  - reindexes incoming data to feature_columns FIRST, before any other
  step. Works identically for a single customer (1-row df) or a batch
  (multi-row df from CSV).
- `predict_single(customer_dict, ...)` - wraps one customer dict into a
  1-row DataFrame, returns {churn_probability, churn_prediction}.
  **Verified:** sample new-customer profile (tenure=1, month-to-month,
  electronic check, no add-ons - the highest-risk EDA/SHAP profile)
  scored 0.9169 / "Yes", as expected.
- `predict_from_csv(csv_path, ..., output_path=None)` - batch version,
  reuses preprocess_customer_data() unchanged. Keeps customerID (and any
  other non-feature columns) in the output for traceability, even though
  they're dropped internally before reaching the model.
- `main()` - CLI entry point (`python src/predict.py customers.csv`).
- **Verified with a 3-row test CSV** covering distinct risk profiles:
  new/month-to-month/no-addons (0.9169, Yes), long-tenure/2yr/full-addons
  (0.1339, No), and a mixed-signal profile (0.8518, Yes) - all landed in
  the expected relative order, and the single-customer test case matched
  its CSV-batch result exactly (0.9169 both times), confirming
  predict_single and predict_from_csv share identical underlying logic.

### Bug found and fixed during development (not in original plan)
- **Column-order fragility:** StandardScaler and LGBMClassifier both
  operate on plain NumPy arrays once data reaches them - features are
  identified by POSITION, not name. An early draft of predict.py built
  the customer DataFrame directly from a caller-supplied dict with no
  guaranteed column order, which would silently feed the model the
  wrong feature in the wrong position if a caller's dict happened to be
  ordered differently than training data - with no error, just a
  confidently wrong prediction. Fixed by loading the canonical column
  order from X_train.columns once (in load_artifacts) and reindexing
  all incoming customer data to that exact order first thing in
  preprocess_customer_data, with a clear ValueError if required columns
  are missing.

### Design decisions made this phase
- **OneHotEncoder fit location (Option B chosen):** fit on X_train only,
  after the split, moved into train.py alongside scale_features() -
  rather than fitting on the full dataset pre-split in
  data_preprocessing.py (Option A). Reasoning: matches the existing
  leakage-prevention principle already established for StandardScaler;
  low practical risk either way for this dataset, but Option B is the
  more defensible/correct pattern to have learned.
- **predict.py input formats:** supports both a single customer dict
  (predict_single) and a batch CSV (predict_from_csv), sharing one core
  preprocessing function. Reasoning: the single-record path is what a
  future Phase 6 FastAPI endpoint will call directly; the CSV path adds
  batch-scoring capability for near-zero extra cost since nothing about
  the core logic needed to change to support both.

## Phase 6 — FastAPI + Docker — IN PROGRESS (api/app.py done; Dockerfile written + commented, not yet built/run)

### api/app.py — built and verified end-to-end against the real champion model
Wraps predict.py's existing predict_single() in a REST API - deliberately
contains NO new prediction logic, purely a translation layer (HTTP
request -> dict -> predict_single() -> dict -> HTTP response), so nothing
already tested in Phase 5/9 gets reimplemented or duplicated.

- `sys.path` manipulation at the top of api/app.py adds `src/` to the
  import path, mirroring the exact same fix tests/conftest.py already
  applies for pytest - api/ and src/ are sibling folders, so Python
  doesn't auto-discover one from the other.
- `CustomerData` (Pydantic model) - mirrors the 19 raw feature columns
  the model was trained on (config.yaml's binary_cols/nominal_cols/
  ordinal_mappings, plus the 4 already-numeric passthrough columns:
  SeniorCitizen, tenure, MonthlyCharges, TotalCharges).
- `PredictionResponse` (Pydantic model) - churn_probability (float) +
  churn_prediction (str), matching predict_single()'s existing return shape.
- `lifespan()` (FastAPI startup/shutdown hook) - loads the champion
  model + encoder + scaler + feature_columns via predict.py's existing
  `load_artifacts()` ONCE when the server starts, stored in a module-level
  `ml_artifacts` dict for every request to reuse. Deliberately NOT loaded
  inside the /predict function itself - would re-read MLflow/joblib
  artifacts from disk on every single request for no benefit, since the
  model never changes between requests.
- `CONFIG_PATH` built from `__file__` (not a relative string) - so
  config.yaml resolves correctly regardless of what directory the
  process is launched from, which matters once this runs inside Docker
  with a different working-directory setup than local dev.
- `/health` - trivial liveness check, no model logic, useful for Docker
  healthchecks later.
- `/predict` - accepts CustomerData, calls predict_single() unchanged,
  returns PredictionResponse.

### Design decision: strict Pydantic validation (Literal types) chosen
Every categorical field (InternetService, Contract, PaymentMethod, etc.)
uses `Literal[...]` instead of plain `str`, so a typo like
`"Fiber Optic"` (wrong case) is rejected at the API boundary with a
clear 422 error - BEFORE it reaches the model. Reasoning: without this,
a typo would silently be treated as an "unknown category" by the
OneHotEncoder (`handle_unknown='ignore'`) and produce a confidently
wrong prediction with no error at all - same failure class as the
column-order bug found and fixed in Phase 5. Encoder's leniency is for
genuinely novel categories in real-world data; Pydantic's strictness is
for catching caller typos - each guards a different failure mode.

### Design decision (deferred to Dockerfile step): how the image gets
model files
mlruns/, mlflow.db, and artifacts/*.joblib are all gitignored, so a
bare `git clone` doesn't include them. Decided to COPY them in from the
local machine at Docker build time (works today, since they already
exist locally from training) rather than setting up a DVC remote +
`dvc pull` inside the build (more "correct" long-term, but a bigger
lift - no remote is configured yet, `.dvc/config` is currently empty -
and out of scope for a portfolio-project Phase 6; revisit only if this
ever needs to run reproducibly in CI or on a machine that never ran
train.py locally).

### Verified end-to-end (real champion model, not synthetic)
Ran `uvicorn api.app:app --reload` locally against the real trained
artifacts:
- Startup logs confirmed the MLflow model download + clean
  "Application startup complete" with no errors.
- `GET /health` -> `{"status":"ok"}` (200 OK).
- `POST /predict` with a real high-risk customer profile -> real
  prediction returned (`churn_probability: 0.9183, churn_prediction:
  "Yes"`) - in the same range as Phase 5's predict.py result (0.9169)
  for a similar profile; not an exact match since the test profile's
  MonthlyCharges/TotalCharges were placeholder values, not the exact
  Phase 5 CSV row. Worth rerunning with the exact Phase 5 row once, to
  confirm an exact match between predict.py and the API.
- `POST /predict` with a deliberately misspelled category
  (`"Fiber Optic"` instead of `"Fiber optic"`) -> correctly rejected
  with a 422 and a clear message naming the field and valid values,
  proving strict validation actually blocks bad input rather than
  silently mispredicting.

### Dockerfile — written and fully commented (not yet built/run)
Built part-by-part with Claude teaching each concept (image vs.
container, layers, caching) before writing the corresponding lines.

- `FROM python:3.13-slim` - matches CI/local Python version exactly;
  slim variant keeps image size down.
- `WORKDIR /app` - anchors all relative paths inside the container to
  `/app`, so config.yaml paths and api/app.py's `__file__`-based
  CONFIG_PATH resolve consistently regardless of launch context.
- `COPY requirements.txt .` then `RUN pip install --no-cache-dir -r
  requirements.txt` done BEFORE copying any code - deliberate layer-
  caching ordering: requirements.txt changes rarely, src/api code
  changes often, so most rebuilds skip re-installing every package
  and only redo the fast "copy code" layers below.
- Explicit per-folder `COPY` lines (`src/`, `api/`, `configs/`,
  `mlruns/`, `mlflow.db`, `artifacts/`) instead of one `COPY . .` -
  keeps caching granular and avoids shipping `.git/`, `notebooks/`,
  `tests/`, `__pycache__/` etc. into the image.
- `COPY mlruns/ ./mlruns/`, `COPY mlflow.db ./mlflow.db`,
  `COPY artifacts/ ./artifacts/` - implements the Phase 6 design
  decision (local COPY at build time instead of DVC remote/`dvc
  pull`). Explicit tradeoff documented in-file: this Dockerfile can
  only build successfully on a machine that already has these three
  gitignored paths populated locally (i.e. one that's already run
  train.py) - a DVC remote is the more "correct" long-term fix,
  deliberately parked for later.
- `EXPOSE 8000` - documentation/contract only, does not itself publish
  the port; that happens at `docker run -p` time.
- `CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port",
  "8000"]` - exec-form array (not shell-string form); `api.app:app`
  resolves via the `api/` folder + `WORKDIR /app`; `--host 0.0.0.0` is
  required (not optional) since the default `127.0.0.1` would only
  accept connections from inside the container itself, causing every
  external request to silently time out.

### Blocked on: Docker Desktop installation (local machine)
- Attempted `docker build -t churn-api .` from repo root - failed,
  Docker not yet installed on the Windows machine
  (`CommandNotFoundException` in PowerShell).
- Local machine also runs VMware Workstation. Checked compatibility:
  modern VMware Workstation (15.5.5+) supports the Windows Hypervisor
  Platform and coexists with WSL2/Hyper-V (which Docker Desktop uses
  on Windows) - some performance overhead possible, but should not be
  a blocker. Proceeding with Docker Desktop install.
- Confirmed target architecture: **amd64** (VMware Workstation itself
  doesn't run on ARM64 Windows, which was enough to confirm without
  needing to check System Settings directly).
- **Next steps once Docker Desktop is installed + verified**
  (`docker --version` and `docker run hello-world` both succeed):
  run `docker build -t churn-api .` from repo root for the first real
  build attempt.

### Not yet done
- Docker Desktop installation + verification on local machine (in
  progress - see "Blocked on" above)
- `docker build` - first real build attempt (Dockerfile is ready, just
  never actually run yet)
- Docker container run + `/health` and `/predict` verification against
  the real champion model (parity check with Phase 6's native
  uvicorn run, which returned churn_probability: 0.9183)

## Phase 9 — Reproducibility + CI — COMPLETE

### tests/test_preprocessing.py — now has real content (closes Phase 5 leftover)
- 6 tests for data_preprocessing.py's pure functions: clean_total_charges
  (blank-space -> 0.0 + no-mutation check), drop_identifier_column,
  encode_binary_columns, encode_ordinal_columns (order preservation),
  encode_target_column, split_features_target (shape/no-leakage/
  reproducibility).
- 2 tests for predict.py's preprocess_customer_data() - these test the
  actual column-order bug found and fixed in Phase 5, not just mapping
  logic, so they carry more regression value than the six above:
  - test_preprocess_customer_data_is_column_order_independent - builds
    one customer as two dicts (correct key order vs. deliberately
    scrambled), asserts identical scaled output. Verified this actually
    catches a regression by temporarily disabling the reindex line in
    predict.py and confirming the test failed, before restoring it.
  - test_preprocess_customer_data_raises_on_missing_field - confirms a
    ValueError naming the missing field is raised when a required
    column is absent from the input.
- Design choice (deliberately parked, not a bug): the column-order test
  fits a tiny real OneHotEncoder/StandardScaler in-memory via a pytest
  fixture, rather than loading the actual joblib artifacts from disk
  via load_preprocessing_artifacts(). Keeps the test fast and
  independent of whether a trained model exists on disk (works on a
  fresh clone before train.py has ever been run) - at the cost of not
  proving the *actual* persisted artifacts behave identically. Revisit
  only if a real-artifact integration test becomes worth the added CI
  complexity.
- All 9 tests verified passing via `python -m pytest tests/ -v`.

### requirements.txt — pinned
- Replaced the unpinned package-name list with exact versions from
  `pip freeze` run in the real project environment (not a fresh
  throwaway install), so the pin reflects what actually trained and
  saved the champion model.
- Fixed two Windows-only transitive dependencies that `pip freeze`
  flattens in unconditionally: `pywin32==312` and `pywinpty==3.0.5`
  both got a `; sys_platform == "win32"` marker added, so pip installs
  them on Windows but skips them cleanly on Linux. `pywinpty` was
  confirmed to hard-fail Linux CI (Rust build error, Windows-only code
  path gated behind `#[cfg(windows)]`) via an actual failed GitHub
  Actions run before the marker was added.

### .github/workflows/tests.yml — added
- Runs on every `push` and `pull_request`, no branch filter.
- `ubuntu-latest` runner, Python 3.13 (matches local dev version
  3.13.5).
- Steps: checkout -> setup Python -> `pip install -r requirements.txt`
  -> `pytest tests/ -v`.
- Deliberately no DVC pull step - none of the current tests touch real
  data or trained artifacts, so a bare checkout is sufficient.
- Confirmed working end-to-end after the pywinpty fix - CI run passed.

## Phase 9.5 — Post-Completion Cleanup + Hygiene Fixes

After Phase 9 was marked complete, Claude did a full read-through of
every non-data file in the repo (config, src/, tests/, CI workflow,
.gitignore, DVC pointer files, README) against this status doc to
check everything documented actually matches the code. Found four
real, previously-undocumented issues, all now resolved:

- **`requirements.txt` was UTF-16-encoded, not UTF-8.** Root cause:
  `pip freeze > requirements.txt` in Windows PowerShell defaults to
  UTF-16 unless `-Encoding utf8` is passed explicitly. It happened to
  still install fine (pip 24+ auto-detects encoding), so CI wasn't
  actually broken, but it was fragile and non-standard. **Fixed** by
  regenerating via `pip freeze | Out-File -Encoding utf8
  requirements.txt` and confirming via byte inspection
  (`Get-Content -Encoding Byte -TotalCount 4` → `239 187 191 97` = a
  UTF-8 BOM followed by the first character - genuine UTF-8 now, not
  UTF-16). The Windows-only `pywin32`/`pywinpty` markers (see Phase 9
  above) had to be re-applied after regenerating, since a fresh
  `pip freeze` overwrites them.
- **`mlflow.set_experiment("churn_prediction")` was hardcoded in
  `train.py`**, even though `config.yaml` already had
  `mlflow.experiment_name: "churn_prediction"` sitting unused right
  next to it - a direct contradiction of the "no MLflow settings
  hardcoded" principle documented in Phase 4. **Fixed** by reading
  `config['mlflow']['experiment_name']` instead.
- **`config.yaml`'s `artifacts.model_dir: "models/"` was dead
  config** - grepped across all of `src/` and confirmed it was never
  referenced anywhere. Root cause: a leftover from before MLflow
  became the actual model-loading path (`load_champion_model()` loads
  from the MLflow artifact store via `runs:/{run_id}/model`, not from
  a local `models/` folder). **Fixed** by deleting the `model_dir` key
  from `config.yaml`, deleting the now-pointless `models/` folder
  (`git rm -r models/`), and removing the matching dead rules
  (`models/*.pkl`, `models/*.joblib`, `models/*.h5`, `models/*.onnx`)
  from `.gitignore`.
- **Evaluation plots (`confusion_matrix.png`, `roc_curve.png`,
  `shap_summary.png`) were fully gitignored via `artifacts/*`**,
  meaning nobody browsing the GitHub repo could see them without
  cloning and re-running the pipeline - a missed opportunity for a
  portfolio project where these are exactly what a recruiter would
  want to see. **Fixed** by creating a separate, tracked `docs/plots/`
  folder (not covered by the `artifacts/*` gitignore rule) and
  copying the three current plots there for direct README embedding
  in Phase 8. Deliberately a manual copy, not automated - these are
  point-in-time snapshots of the current champion model, so if the
  champion is ever retrained/replaced, `docs/plots/` needs to be
  manually refreshed and recommitted, or it will silently go stale
  relative to the live model. Revisit only if this drift becomes an
  actual problem worth automating around.

## Open TODOs (flagged, not yet fixed - relevant for later phases)

**[Resolved in Phase 5]** encode_nominal_columns (pd.get_dummies) removed
from data_preprocessing.py entirely. Replaced with encode_nominal_features()
in train.py using sklearn.preprocessing.OneHotEncoder, fit on X_train only
(after the split - Option B, matching StandardScaler's leakage-prevention
pattern) with drop='first' (matches old drop_first=True) and
handle_unknown='ignore' (so an unseen category at prediction time encodes
as all-zeros instead of crashing). Fitted encoder saved via joblib to
config's artifacts.preprocessor_path. Verified: all 8 MLflow re-run metrics
matched the pre-refactor numbers exactly, confirming mathematical
equivalence to the old get_dummies pipeline.

**[Resolved in Phase 5]** StandardScaler now saved via joblib to a
dedicated config path (artifacts.scaler_path, separate from
preprocessor_path) inside train.py's main(), right after fitting.
evaluate.py updated to load both the encoder and scaler via joblib
(load_preprocessing_artifacts()) instead of re-fitting them - closes the
"re-fit vs load" inconsistency noted when this TODO was first resolved
partially. apply_preprocessing() (in evaluate.py) applies the loaded
encoder/scaler without fitting, and is reused unchanged by predict.py.

**[Resolved in Phase 5]** Fixed via tests/conftest.py, which pytest
auto-loads before any test in that folder and which adds src/ to
Python's module search path. Verified with
`python -m pytest tests/ --collect-only` (succeeded with "no tests
collected", zero import errors) before any real test content was
written - confirms the path fix works independent of test content.

**[Minor, cosmetic]** `evaluate.py` prints a sklearn UserWarning
("X does not have valid feature names, but LGBMClassifier was fitted
with feature names") because `scale_features()`'s StandardScaler
strips column names from the DataFrame, returning a plain NumPy array.
Harmless — column order is preserved so predictions are still correct.
Optional fix later: `StandardScaler.set_output(transform="pandas")`.

**[Resolved in Phase 4]** SHAP's `TreeExplainer.shap_values()` on newer
SHAP versions returns a list of two arrays for binary classifiers
(`[class_0_values, class_1_values]`) instead of one array. Fixed in
`compute_shap_values()` via an `isinstance(shap_values, list)` check
selecting index `[1]` (Churn class). Documenting here since it's a
version-compatibility gotcha that could resurface if SHAP changes
behavior again, or if this pattern is reused elsewhere (e.g. a future
predict.py explainability feature).

## Not yet started (later phases)
- Phase 6: Dockerfile is written and commented (see Phase 6 section
  above) but not yet built or run - waiting on Docker Desktop install
  on the local machine, then `docker build` + container verification
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
  - **Model blind spots are findings, not bugs:** false-negative analysis
  (Phase 4) showed the champion systematically misses churners with
  "safe" profiles (long tenure, long contract, has add-ons) — this is
  an honest dataset limitation (no signal for price sensitivity,
  competitor offers, support experience), not something to silently
  patch. Worth stating explicitly rather than only reporting aggregate
  recall.
- **Frequency of use ≠ size of impact:** LightGBM's built-in
  `feature_importances_` (split-count) and SHAP's mean |impact| can
  rank features differently even when they agree on the top set —
  cross-checking both gives a more credible "what matters" story than
  relying on a single importance method.
- **Column order is a silent failure mode, not just a style concern:**
  once data reaches StandardScaler/LGBMClassifier as a NumPy array,
  features are identified by position only - a caller-supplied dict
  or CSV in a different column order than training would produce a
  confidently wrong prediction with no error. The fix (predict.py,
  Phase 5) is to load and enforce the canonical training-time column
  order explicitly, rather than assuming input data arrives correctly
  ordered.
- **A function's true test is being called from somewhere new:**
  apply_preprocessing() (evaluate.py, Phase 4) was designed for reuse in
  predict.py "on paper" - Phase 5 confirmed it actually worked unchanged
  when a second caller (predict.py) used it, validating the reuse
  design rather than just assuming it.
- **pip freeze flattens away platform context:** running `pip freeze`
  on Windows bakes in Windows-only transitive dependencies (pywin32,
  pywinpty) with no indication they're conditional - PEP 508
  environment markers (`; sys_platform == "win32"`) have to be added
  back in by hand for the pinned file to install cleanly on Linux CI.
- **A regression test should be proven to fail, not just proven to
  pass:** for the column-order test (Phase 9), the reindex line in
  predict.py was temporarily disabled to confirm the test actually
  fails without the fix, before trusting that it protects anything.
- **PowerShell's `>` redirection defaults to UTF-16, not UTF-8:** any
  command piped into plain `> file.txt` (e.g. `pip freeze >
  requirements.txt`) on Windows PowerShell silently produces a
  UTF-16-encoded file. Tools that expect plain text (pip, most Linux
  CLI tools, some editors) may not handle this consistently - use
  `| Out-File -Encoding utf8` explicitly instead of `>` when the
  output needs to be portable.
- **A "finished" phase is worth a second read-through pass:** the
  Phase 9.5 cleanup (dead config, hardcoded experiment name, file
  encoding, untracked evaluation plots) was only found by re-reading
  every file against project_status.md after Phase 9 was already
  marked complete - none of these were caught during the phase itself.
  Worth doing this kind of pass occasionally rather than only trusting
  in-the-moment verification.