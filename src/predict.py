"""
predict.py

Loads the persisted champion model, encoder, and scaler, then applies
them to new, unseen customer data to predict churn - either a single
customer (as a dict) or a batch (as a CSV file).

Design principle (matches train.py/evaluate.py): this script never
fits anything. Everything it uses (model, encoder, scaler) was already
fit during training and saved to disk - a single new customer record
has no "training data" of its own to fit a fresh encoder/scaler on, so
loading persisted artifacts isn't just a nice-to-have here, it's the
only correct option.
"""

import pandas as pd
import mlflow
import mlflow.lightgbm
import joblib
from data_preprocessing import (
    load_config,
    clean_total_charges,
    encode_binary_columns,
    encode_ordinal_columns,
)
from evaluate import load_preprocessing_artifacts, apply_preprocessing


def load_champion_model_from_export(config):
    """
    Load the champion model from its self-contained exported copy
    (config['artifacts']['champion_model_dir']) instead of through
    MLflow's tracking store (runs:/{run_id}/model).

    evaluate.py's load_champion_model() resolves the model through
    mlflow.db, which records the ABSOLUTE local filesystem path to
    each run's artifacts from the machine train.py was run on - fine
    for local evaluation, but broken inside Docker, where that path
    doesn't exist. This function instead loads directly from a plain
    local folder (produced once by export_champion_model.py, which
    must be rerun whenever champion_run_id changes) - just files on
    disk, no tracking-store lookup, portable to any machine/container.

    Deliberately kept separate from evaluate.py's load_champion_model()
    rather than replacing it: evaluate.py's local workflow (SHAP plots,
    confusion matrix, etc.) is meant to reflect the exact tracked run
    by run_id, while predict.py (used by both the CLI and the Docker
    API) only needs a working, loadable copy of that same model.
    """
    model_dir = config['artifacts']['champion_model_dir']
    model = mlflow.lightgbm.load_model(model_dir)
    return model


def load_artifacts(config):
    """
    Load everything predict.py needs, once: the champion model (from
    its exported local folder - see load_champion_model_from_export),
    the fitted encoder/scaler (from joblib), and the exact raw feature
    column order training used (from a small joblib artifact).

    That column order is loaded here - not hardcoded, not assumed from
    whatever order a caller's dict happens to be in - because
    StandardScaler and LGBMClassifier both work on plain NumPy arrays
    internally, which only know features by POSITION, not by name.
    A new customer's data must be forced into this exact same column
    order before anything else, or the model would silently score the
    wrong feature in the wrong position with no error at all.

    Note: no mlflow.set_tracking_uri() call here anymore - none of
    these three loads (model, encoder/scaler, feature_columns) touch
    the tracking store; they're all plain local files now. Setting a
    tracking URI that's never used would just be dead code.
    """
    model = load_champion_model_from_export(config)
    encoder, scaler = load_preprocessing_artifacts(config)
    feature_columns = joblib.load(config['artifacts']['feature_columns_path'])

    return model, encoder, scaler, feature_columns


def preprocess_customer_data(df, config, encoder, scaler, feature_columns):
    """
    Apply the full training-time preprocessing pipeline to new customer
    data - cleaning, binary/ordinal encoding, then the already-fitted
    OneHotEncoder + StandardScaler. Works for both a single customer
    (1-row DataFrame) and a batch (multi-row DataFrame from a CSV) -
    same steps either way, since nothing here is fit fresh.

    Reindexes to feature_columns FIRST, before any other processing.
    This is what guarantees correct column order regardless of what
    order the caller's data arrived in, and it also handles dropping
    customerID (or any other extra column) automatically, since only
    columns present in feature_columns survive the reindex. If a
    required column is genuinely missing from the input, it becomes a
    clear, named error here instead of a silent misalignment later.
    """
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required customer fields: {sorted(missing)}")

    df = df.reindex(columns=feature_columns)

    df = clean_total_charges(df)
    df = encode_binary_columns(df, config['features']['binary_cols'])
    df = encode_ordinal_columns(df, config['features']['ordinal_mappings'])

    X_encoded, X_scaled = apply_preprocessing(
        df, encoder, scaler, config['features']['nominal_cols']
    )
    return X_scaled


def predict_single(customer_dict, model, encoder, scaler, feature_columns, config):
    """
    Predict churn for one new customer, given as a plain Python dict
    (e.g. {'gender': 'Female', 'tenure': 5, 'Contract': 'Month-to-month', ...}).
    Key order in the dict doesn't matter - preprocess_customer_data
    reindexes to the correct order regardless.

    Returns a dict with the predicted churn probability (0-1) and the
    final Yes/No label at the default 0.5 threshold - the same
    threshold train.py's evaluate_model() and the confusion matrix in
    evaluate.py both used, so this stays consistent with how the model
    was already evaluated.
    """
    df = pd.DataFrame([customer_dict])
    X_scaled = preprocess_customer_data(df, config, encoder, scaler, feature_columns)

    churn_probability = model.predict_proba(X_scaled)[0, 1]
    churn_prediction = "Yes" if churn_probability >= 0.5 else "No"

    return {
        "churn_probability": round(float(churn_probability), 4),
        "churn_prediction": churn_prediction,
    }


def predict_from_csv(csv_path, model, encoder, scaler, feature_columns, config, output_path=None):
    """
    Predict churn for every customer row in a CSV file (batch scoring).

    Reuses preprocess_customer_data() unchanged - it already works on
    any number of rows, since nothing in it assumes a single customer.
    The original dataframe (df) is kept alongside the predictions when
    building `results`, so columns like customerID survive into the
    output even though they get dropped internally when reindexing to
    feature_columns for the model - useful for tracing which prediction
    belongs to which customer, without customerID ever being fed into
    the model itself.

    If output_path is given, saves the results to a new CSV; either
    way, always returns the results DataFrame.
    """
    df = pd.read_csv(csv_path)
    X_scaled = preprocess_customer_data(df, config, encoder, scaler, feature_columns)

    churn_probabilities = model.predict_proba(X_scaled)[:, 1]
    churn_predictions = ["Yes" if p >= 0.5 else "No" for p in churn_probabilities]

    results = df.copy()
    results["churn_probability"] = churn_probabilities.round(4)
    results["churn_prediction"] = churn_predictions

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return results


def main():
    """
    Command-line entry point for batch prediction:
        python src/predict.py path/to/customers.csv

    Loads all artifacts once, scores every row in the given CSV, and
    saves results next to the input file (same name + "_predictions"
    suffix), so predict.py is usable both as an importable module
    (predict_single/predict_from_csv, e.g. from a future FastAPI
    endpoint) and as a standalone script for one-off batch scoring.
    """
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/customers.csv")
        return

    csv_path = sys.argv[1]
    output_path = csv_path.replace(".csv", "_predictions.csv")

    config = load_config()
    model, encoder, scaler, feature_columns = load_artifacts(config)

    results = predict_from_csv(csv_path, model, encoder, scaler, feature_columns, config, output_path)
    print(f"\nScored {len(results)} customers.")
    print(results[["churn_probability", "churn_prediction"]].head())


if __name__ == "__main__":
    main()