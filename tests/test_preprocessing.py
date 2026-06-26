"""
Sanity test for the data preprocessing pipeline.

Run from the project root using:

    python tests/test_preprocessing.py
"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import (
    load_config,
    load_raw_data,
    clean_total_charges,
    drop_identifier_column,
    encode_binary_columns,
    encode_ordinal_columns,
    encode_nominal_columns,
)


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    # -------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------
    config = load_config()
    print("\n✓ Configuration loaded successfully.")

    # -------------------------------------------------------
    # Load raw dataset
    # -------------------------------------------------------
    df = load_raw_data(config["data"]["raw_path"])

    print("\n[1] RAW DATA")
    print("-" * 70)
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(df.head(3))

    # -------------------------------------------------------
    # Clean TotalCharges
    # -------------------------------------------------------
    df = clean_total_charges(df)

    print("\n[2] CLEAN TOTALCHARGES")
    print("-" * 70)
    print("dtype:", df["TotalCharges"].dtype)
    print("Missing values:", df["TotalCharges"].isnull().sum())

    # -------------------------------------------------------
    # Drop customerID
    # -------------------------------------------------------
    before_cols = df.shape[1]

    df = drop_identifier_column(df)

    after_cols = df.shape[1]

    print("\n[3] DROP IDENTIFIER COLUMN")
    print("-" * 70)
    print(f"Columns: {before_cols} → {after_cols}")
    print("customerID exists:", "customerID" in df.columns)

    # -------------------------------------------------------
    # Binary Encoding
    # -------------------------------------------------------
    df = encode_binary_columns(
        df,
        config["features"]["binary_cols"],
    )

    print("\n[4] BINARY ENCODING")
    print("-" * 70)

    for col in config["features"]["binary_cols"]:
        print(f"{col:20} -> {sorted(df[col].unique())}")

    # -------------------------------------------------------
    # Ordinal Encoding
    # -------------------------------------------------------
    df = encode_ordinal_columns(
        df,
        config["features"]["ordinal_mappings"],
    )

    print("\n[5] ORDINAL ENCODING")
    print("-" * 70)

    print("Contract value counts:")
    print(df["Contract"].value_counts().sort_index())

    # -------------------------------------------------------
    # One-Hot Encoding
    # -------------------------------------------------------
    before_shape = df.shape

    df = encode_nominal_columns(
        df,
        config["features"]["nominal_cols"],
    )

    after_shape = df.shape

    print("\n[6] NOMINAL ENCODING")
    print("-" * 70)

    print(f"Shape: {before_shape} → {after_shape}")

    object_cols = df.select_dtypes(include="object").columns.tolist()

    print("\nRemaining object columns:")
    print(object_cols)

    # -------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------
    print("\n[7] FINAL DATASET")
    print("-" * 70)

    print(df.head())

    print("\nFinal shape:", df.shape)

    print("\nData types:")
    print(df.dtypes.value_counts())

    print("\n✓ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()