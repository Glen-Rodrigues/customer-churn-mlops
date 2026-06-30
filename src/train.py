"""
train.py

Trains churn prediction models on the preprocessed Telco data and tracks
experiments with MLflow.

Design principle: all ML logic here (loading data, fitting models,
computing metrics) is kept MLflow-agnostic - these functions don't know
or care that MLflow exists, so they're easy to test and reuse elsewhere
(e.g. in predict.py or a notebook). The MLflow logging calls live only
in main(), which acts as the orchestrator: it calls the pure functions,
gets results back, and logs whatever it wants to MLflow.

This mirrors the structure of data_preprocessing.py - small, focused,
testable functions chained together by a single entry point.
"""

import pandas as pd
import os
from data_preprocessing import load_config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


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


def main():
    """
    Orchestrate the full training pipeline for one model run and track it
    with MLflow.

    This is the only function in this file that knows MLflow exists - it
    loads config and data, calls the pure ML functions (train, scale,
    evaluate) to get back a model and metrics, then logs the run's
    params, metrics, and the model artifact to MLflow. Keeping all
    mlflow.* calls confined to main() is what keeps the functions above
    MLflow-agnostic, testable, and reusable elsewhere.
    """
    config = load_config()

    X_train, X_test, y_train, y_test = load_processed_data(
        config['data']['processed_dir'],
        config['data']['target_column']
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    params = {"max_iter": 1000, "random_state": 42}

    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run():
        model = train_logistic_regression(X_train_scaled, y_train, **params)

        metrics = evaluate_model(model, X_test_scaled, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(metrics)


if __name__ == "__main__":
    main()