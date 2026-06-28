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


if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test = load_processed_data(
        config['data']['processed_dir'],
        config['data']['target_column']
    )
    
    model = train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42)
    print("Model trained successfully:", model)