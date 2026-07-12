"""
train.py

Trains churn prediction models on the preprocessed Telco data and tracks
experiments with MLflow.

Design principle: all ML logic here (loading data, fitting models,
computing metrics) is kept MLflow-agnostic - these functions don't know
or care that MLflow exists, so they're easy to test and reuse elsewhere
(e.g. in predict.py or a notebook). MLflow logging is confined to
run_experiment(), which trains/evaluates/logs a single run for a given
model-training function, its matching MLflow log_model function, and a
hyperparameter dict. Both the training function (train_fn) and the
logging function (log_fn) are passed into run_experiment() rather than
hardcoded (dependency injection) - this is what lets one function
support Logistic Regression, XGBoost, and LightGBM interchangeably,
without duplicating the run-train-log sequence per model family.

main() is the top-level orchestrator: it loads and scales the data once,
computes the class-imbalance ratio (scale_pos_weight) once from y_train,
then calls run_experiment() once per model/hyperparameter configuration
being compared - covering three model families (LR, XGBoost, LightGBM),
each with a default and imbalance-corrected run, plus a tuned run for
the two tree models. All 8 runs land in the same MLflow experiment for
side-by-side comparison.

This mirrors the structure of data_preprocessing.py - small, focused,
testable functions chained together by orchestrator function(s).
"""

import pandas as pd
import os
from data_preprocessing import load_config
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.preprocessing import OneHotEncoder
import joblib


def load_processed_data(processed_dir, target_column):
    """
    Load train.csv and test.csv from processed_dir, split each into
    features (X) and target (y).
    
    Returns: X_train, X_test, y_train, y_test
    """
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]

    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Train a Logistic Regression model on the training data.

    Hyperparameters (e.g. max_iter, random_state, class_weight) are passed
    in via **kwargs rather than hardcoded, so different configurations can
    be tried from the call site (or later, MLflow experiment configs)
    without editing this function.
    """
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train, y_train, **kwargs):
    """
    Train an XGBoost classifier on the training data.

    Same shape as train_logistic_regression (X_train, y_train, **kwargs)
    -> fitted model - this consistent shape is what lets run_experiment()
    accept it interchangeably as train_fn. XGBoost builds trees
    sequentially, each correcting errors from the previous ones (gradient
    boosting), which can capture non-linear relationships and feature
    interactions that Logistic Regression can't (relevant here given
    EDA found Contract type and InternetService confounded with
    each other).
    """
    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    return model


def train_lightgbm(X_train, y_train, **kwargs):
    """
    Train a LightGBM classifier on the training data.

    Same shape as train_logistic_regression/train_xgboost
    (X_train, y_train, **kwargs) -> fitted model, so run_experiment()
    can use it interchangeably as train_fn. LightGBM is also a gradient
    boosted trees model (like XGBoost), but grows trees leaf-wise
    instead of level-wise, which is typically faster and can be more
    accurate - included here to complete the planned comparison across
    model families (LR vs XGBoost vs LightGBM), not just hyperparameter
    variants of one model.
    """
    model = LGBMClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    return model


def encode_nominal_features(X_train, X_test, nominal_cols):
    """
    One-hot encode nominal (unordered, 3+ category) columns using
    sklearn's OneHotEncoder instead of pd.get_dummies.

    Fit only on X_train's nominal columns, then used to transform both
    X_train and X_test - same leakage-prevention principle as
    scale_features() (test set categories don't influence the fit).
    Fitting once (vs pd.get_dummies deriving columns fresh from
    whatever data it's given) also means this encoder can be saved and
    reused in predict.py, where a single new customer record won't have
    every category present - get_dummies would produce a different/
    wrong column set than training in that case.

    drop='first' matches the original pd.get_dummies(drop_first=True)
    behavior (avoids the dummy variable trap). handle_unknown='ignore'
    means a category never seen during training won't crash predict.py
    later - it just encodes as all-zeros for that customer instead.
    """
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    X_train_nominal = encoder.fit_transform(X_train[nominal_cols])
    X_test_nominal = encoder.transform(X_test[nominal_cols])

    encoded_cols = encoder.get_feature_names_out(nominal_cols)
    X_train_nominal_df = pd.DataFrame(X_train_nominal, columns=encoded_cols, index=X_train.index)
    X_test_nominal_df = pd.DataFrame(X_test_nominal, columns=encoded_cols, index=X_test.index)

    X_train_encoded = pd.concat([X_train.drop(columns=nominal_cols), X_train_nominal_df], axis=1)
    X_test_encoded = pd.concat([X_test.drop(columns=nominal_cols), X_test_nominal_df], axis=1)

    return X_train_encoded, X_test_encoded, encoder


def scale_features(X_train, X_test):
    """
    Scale numeric features using StandardScaler (mean=0, std=1).

    The scaler is fit only on X_train, then used to transform both
    X_train and X_test - fitting on test data (or on combined data)
    would leak test set statistics into training, inflating performance
    estimates. The fitted scaler is returned so the exact same
    transformation can be reused later (e.g. in predict.py on new,
    unseen customer records).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on the test set.

    Computes accuracy, precision, recall, F1, and ROC-AUC. Accuracy alone
    is misleading here given the ~73/27 class imbalance found in EDA (a
    model predicting "No churn" for everyone would score ~73.5% accuracy
    while being useless) - the other metrics, especially recall, matter
    more since missing an actual churner is typically costlier than a
    false alarm.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics


def run_experiment(train_fn, log_fn, params, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Run one full MLflow-tracked training experiment for ANY supported
    model type: train a model using train_fn with the given
    hyperparameters, evaluate it on the test set, and log
    params/metrics/model as a single MLflow run.

    train_fn (e.g. train_logistic_regression / train_xgboost /
    train_lightgbm) and log_fn (e.g. mlflow.sklearn.log_model /
    mlflow.xgboost.log_model / mlflow.lightgbm.log_model) are both
    passed in rather than hardcoded, so this one function can run
    experiments across different model families - each library needs
    its own MLflow log_model flavor, since e.g. mlflow.sklearn.log_model
    can't safely serialize XGBoost's native types.

    Pulled out as its own function so main() can call it multiple times
    with different train_fn/log_fn/params combinations (e.g. LR baseline
    vs LR balanced vs XGBoost vs LightGBM) without duplicating the
    training/logging logic for each comparison.
    """
    with mlflow.start_run():
        model = train_fn(X_train_scaled, y_train, **params)

        metrics = evaluate_model(model, X_test_scaled, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        log_fn(model, "model")

        print("\nParameters:", params)
        print("Metrics:", metrics)


def main():
    """
    Orchestrate the full Phase 3 model comparison and track every
    experiment with MLflow.

    Loads and scales the data once, computes the class-imbalance ratio
    (scale_pos_weight = negative_count / positive_count, from y_train)
    once, then runs 8 MLflow-tracked experiments via run_experiment():

    1. Logistic Regression baseline
    2. Logistic Regression with class_weight='balanced'
    3. XGBoost baseline (default hyperparameters)
    4. XGBoost with scale_pos_weight (imbalance-corrected)
    5. LightGBM baseline (default hyperparameters)
    6. LightGBM with scale_pos_weight (imbalance-corrected)
    7. XGBoost tuned (scale_pos_weight + n_estimators/max_depth/
       learning_rate hand-picked for tabular data this size)
    8. LightGBM tuned (same tuning as #7)

    This compares both hyperparameter variants of one model (imbalance
    handling) AND whole model families (LR vs XGBoost vs LightGBM),
    per the original Phase 3 scope. All runs land in the same MLflow
    experiment ("churn_prediction") for side-by-side comparison.

    Also fits the nominal-column OneHotEncoder and StandardScaler here
    (on X_train only, in that order) and saves both fitted objects via
    joblib to config's artifacts.preprocessor_path / scaler_path. This
    is what makes predict.py possible later - a single new customer
    record has no training data to fit a fresh encoder/scaler on, so
    the exact fitted objects from this training run must be persisted
    and reloaded at inference time rather than recomputed

    Champion model (selected after comparing all 8 runs): LightGBM
    Tuned - best ROC-AUC and F1 of all runs, with recall close to the
    single highest recall achieved (LR balanced). See run #8's params
    below (lgbm_tuned_params) for the exact winning configuration.
    """
    config = load_config()

    # Point at SQLite backend from config, so train.py and evaluate.py always agree on run storage
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    X_train, X_test, y_train, y_test = load_processed_data(
        config['data']['processed_dir'],
        config['data']['target_column']
    )

    X_train_encoded, X_test_encoded, encoder = encode_nominal_features(
        X_train, X_test, config['features']['nominal_cols']
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_encoded, X_test_encoded)

    # Persist the fitted encoder and scaler so predict.py can apply the
    # exact same transformations to a single new customer record later,
    # instead of re-deriving them (which isn't possible with only one row).
    os.makedirs(os.path.dirname(config['artifacts']['preprocessor_path']), exist_ok=True)
    joblib.dump(encoder, config['artifacts']['preprocessor_path'])
    joblib.dump(scaler, config['artifacts']['scaler_path'])
    print(f"Encoder saved to: {config['artifacts']['preprocessor_path']}")
    print(f"Scaler saved to: {config['artifacts']['scaler_path']}")
    
    # --- Logistic Regression: baseline vs class-imbalance corrected ---
    baseline_params = {"max_iter": 1000, "random_state": 42}
    balanced_params = {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"}

    # --- XGBoost: baseline (no imbalance handling yet) ---
    xgb_params = {"random_state": 42}

    # Compute the imbalance ratio directly from y_train, so it always reflects
    # the actual current split rather than a hardcoded guess. Shared by
    # XGBoost and LightGBM below, since both use the same scale_pos_weight
    # convention (unlike LR's class_weight='balanced' string shortcut).
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # --- XGBoost: imbalance corrected ---
    xgb_balanced_params = {"random_state": 42, "scale_pos_weight": scale_pos_weight}

    # --- LightGBM: baseline vs imbalance corrected ---
    lgbm_params = {"random_state": 42}
    lgbm_balanced_params = {"random_state": 42, "scale_pos_weight": scale_pos_weight}

    # Quick hand-picked tuning for the tree models, combined with the
    # scale_pos_weight imbalance fix already shown to help recall.
    # n_estimators up + learning_rate down = more, smaller correction
    # steps (generally more stable). max_depth capped at 4 to avoid
    # overfitting on a dataset this size (~7000 rows).
    xgb_tuned_params = {
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
    }

    # lgbm_tuned_params is the CHAMPION model's exact configuration,
    # selected after comparing all 8 runs (see docstring above and
    # project_status.md for the full reasoning/metrics table).
    lgbm_tuned_params = {
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
    }

    mlflow.set_experiment("churn_prediction")
    run_experiment(train_logistic_regression, mlflow.sklearn.log_model, baseline_params, X_train_scaled, y_train, X_test_scaled, y_test)
    run_experiment(train_logistic_regression, mlflow.sklearn.log_model, balanced_params, X_train_scaled, y_train, X_test_scaled, y_test)
    run_experiment(train_xgboost, mlflow.xgboost.log_model, xgb_params, X_train_scaled, y_train, X_test_scaled, y_test)
    run_experiment(train_xgboost, mlflow.xgboost.log_model, xgb_balanced_params, X_train_scaled, y_train, X_test_scaled, y_test)
    run_experiment(train_lightgbm, mlflow.lightgbm.log_model, lgbm_params, X_train_scaled, y_train, X_test_scaled, y_test)
    run_experiment(train_lightgbm, mlflow.lightgbm.log_model, lgbm_balanced_params, X_train_scaled, y_train, X_test_scaled, y_test)
    run_experiment(train_xgboost, mlflow.xgboost.log_model, xgb_tuned_params, X_train_scaled, y_train, X_test_scaled, y_test)
    run_experiment(train_lightgbm, mlflow.lightgbm.log_model, lgbm_tuned_params, X_train_scaled, y_train, X_test_scaled, y_test)


if __name__ == "__main__":
    main()