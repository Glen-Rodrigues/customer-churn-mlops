"""
data_preprocessing.py

Takes raw Telco churn data and prepares it for modeling:
- Fixes data quality issues (TotalCharges)
- Encodes categorical features
- Splits into train/test sets

Designed as reusable functions so predict.py can apply the same
cleaning/encoding logic to new, single customer records later.
"""

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path="configs/config.yaml"):
    """Load settings (paths, split params, feature lists) from the YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(raw_path):
    """Load the raw Telco churn CSV from disk."""
    df = pd.read_csv(raw_path)
    return df


def clean_total_charges(df):
    """
    Fix TotalCharges: it's loaded as a string because 11 rows contain a
    blank space instead of a number (these are tenure=0 new customers).
    Convert to numeric, and fill those blanks with 0 since brand-new
    customers genuinely haven't accrued charges yet.
    """
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df


def drop_identifier_column(df):
    """
    Drop customerID - it's a unique identifier, not a predictive feature.
    The model should never see this; it has no relationship to churn,
    it's just a label for the row.
    """
    df = df.copy()
    df = df.drop('customerID', axis=1)
    return df


def encode_binary_columns(df, binary_cols):
    """
    Map Yes/No binary columns to 1/0.
    Used for columns with exactly two categories where there's no
    risk of implying false order (e.g. gender, Partner, Dependents).
    """
    df = df.copy()
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    return df


def encode_ordinal_columns(df, ordinal_mappings):
    """
    Map ordered categorical columns to integers reflecting their real order.

    ordinal_mappings is a dict where each key is a column name and each
    value is itself a dict mapping category -> integer, e.g.:
        {'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}}

    This way the function isn't hardcoded to Contract specifically -
    any ordinal column can be added just by extending the dict passed in.
    """
    df = df.copy()
    for col, order_map in ordinal_mappings.items():
        df[col] = df[col].map(order_map)
    return df


def encode_nominal_columns(df, nominal_cols):
    """
    One-hot encode nominal columns (3+ categories, no natural order).
    drop_first=True avoids the dummy variable trap - if all the
    remaining one-hot columns are 0, the model can infer the dropped
    category, so we don't need it explicitly.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    return df


def encode_target_column(df, target_column):
    """
    Map the target column (Churn) from Yes/No strings to 1/0.
    Kept separate from encode_binary_columns since the target plays a
    fundamentally different role in modeling than input features -
    it's the answer, not a predictor.
    """
    df = df.copy()
    df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})
    return df


def split_features_target(df, target_column, test_size, random_state):
    """
    Separate the dataframe into X (features) and y (target), then split
    each into train and test sets.

    X = everything the model is allowed to see when making a prediction.
    y = the actual answer (Churn) the model is trying to learn to predict.

    test_size and random_state come from config.yaml so the split is
    reproducible - the same random_state always produces the same split,
    which matters for fair comparison between different model runs later.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, target_column, output_dir):
    """
    Save the train/test split to disk as two CSV files: train.csv and test.csv.

    Each file recombines the features (X) with the target (y) into one
    dataframe before saving - this mirrors how the raw data was originally
    shaped, and means train.py only needs to load two files and can split
    X/y again itself using the same target_column name from config.

    output_dir is the folder to save into (created if it doesn't exist).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Recombine features and target back into one dataframe per split
    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path