"""
test_preprocessing.py

Unit tests for the pure functions in src/data_preprocessing.py, plus
the column-order-safety tests for predict.py's preprocess_customer_data().

These use small, hand-built DataFrames instead of the real CSV -
that's the point of a unit test: we know exactly what goes in, so we
can assert exactly what should come out, with no ambiguity about
whether a bug is hiding somewhere in 7000 real rows.

conftest.py (already in this folder) adds src/ to the import path,
which is why these plain imports work even though this file lives in
tests/, not src/.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import (
    clean_total_charges,
    drop_identifier_column,
    encode_binary_columns,
    encode_ordinal_columns,
    encode_target_column,
    split_features_target,
)
from predict import preprocess_customer_data


def test_clean_total_charges_converts_blank_space_to_zero():
    """
    The real dataset has 11 rows where TotalCharges is a single blank
    space (' ') instead of a number - all new customers with tenure=0.
    This test recreates that exact scenario on a tiny DataFrame and
    checks the fix does what we decided in Phase 1: blank -> 0.0,
    not the column mean.
    """
    df = pd.DataFrame({
        'tenure': [0, 5, 12],
        'TotalCharges': [' ', '350.5', '840.0'],  # note: strings, like the real CSV
    })

    result = clean_total_charges(df)

    assert result['TotalCharges'].iloc[0] == 0.0
    assert result['TotalCharges'].iloc[1] == 350.5
    assert result['TotalCharges'].iloc[2] == 840.0
    assert pd.api.types.is_numeric_dtype(result['TotalCharges'])


def test_clean_total_charges_does_not_mutate_input():
    """
    clean_total_charges (like the other functions here) does df.copy()
    internally. This test checks that promise holds - calling the
    function should never change the caller's original DataFrame.
    """
    df = pd.DataFrame({'tenure': [0], 'TotalCharges': [' ']})
    original = df.copy()

    clean_total_charges(df)

    pd.testing.assert_frame_equal(df, original)


def test_drop_identifier_column_removes_customer_id_only():
    """
    customerID should be dropped; every other column should survive
    untouched.
    """
    df = pd.DataFrame({
        'customerID': ['1001-ABC', '1002-DEF'],
        'gender': ['Male', 'Female'],
        'tenure': [5, 10],
    })

    result = drop_identifier_column(df)

    assert 'customerID' not in result.columns
    assert list(result.columns) == ['gender', 'tenure']


def test_encode_binary_columns_maps_yes_no_and_gender():
    """
    Checks the Yes/No -> 1/0 and Male/Female -> 1/0 mapping across
    several binary columns at once.
    """
    df = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'Partner': ['Yes', 'No'],
        'PhoneService': ['No', 'Yes'],
    })

    result = encode_binary_columns(df, binary_cols=['gender', 'Partner', 'PhoneService'])

    assert result['gender'].tolist() == [1, 0]
    assert result['Partner'].tolist() == [1, 0]
    assert result['PhoneService'].tolist() == [0, 1]


def test_encode_ordinal_columns_preserves_order():
    """
    Contract is the key ordinal feature. This checks Month-to-month/One
    year/Two year map to 0/1/2 - i.e. the *order* is preserved.
    """
    df = pd.DataFrame({
        'Contract': ['Two year', 'Month-to-month', 'One year'],
    })
    ordinal_mappings = {
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    }

    result = encode_ordinal_columns(df, ordinal_mappings)

    assert result['Contract'].tolist() == [2, 0, 1]


def test_encode_target_column_maps_churn_to_binary():
    """
    Churn Yes/No -> 1/0. This is the label the whole model is trained
    to predict, so a flipped mapping here would silently invert every
    prediction without any error being raised anywhere.
    """
    df = pd.DataFrame({'Churn': ['Yes', 'No', 'No', 'Yes']})

    result = encode_target_column(df, target_column='Churn')

    assert result['Churn'].tolist() == [1, 0, 0, 1]


def test_split_features_target_shapes_and_no_leakage():
    """
    Three things worth checking about the train/test split:
    1. Row counts respect test_size.
    2. X never contains the target column.
    3. random_state actually makes the split reproducible.
    """
    df = pd.DataFrame({
        'tenure': range(20),
        'MonthlyCharges': range(20),
        'Churn': [0, 1] * 10,
    })

    X_train, X_test, y_train, y_test = split_features_target(
        df, target_column='Churn', test_size=0.2, random_state=42
    )

    assert X_train.shape[0] == 16
    assert X_test.shape[0] == 4
    assert y_train.shape[0] == 16
    assert y_test.shape[0] == 4

    assert 'Churn' not in X_train.columns
    assert 'Churn' not in X_test.columns

    X_train2, X_test2, y_train2, y_test2 = split_features_target(
        df, target_column='Churn', test_size=0.2, random_state=42
    )
    pd.testing.assert_frame_equal(X_train, X_train2)
    pd.testing.assert_frame_equal(X_test, X_test2)


# ---------------------------------------------------------------------------
# Tests for predict.py's preprocess_customer_data()
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_pipeline_pieces():
    """
    Build tiny, real, fitted OneHotEncoder + StandardScaler - the same
    two objects predict.py normally loads from disk via joblib - without
    touching any real files. This is what makes the test below a true
    *unit* test: it exercises the real sklearn transform code path, but
    doesn't depend on your actual trained artifacts existing on disk.

    Returns everything preprocess_customer_data() needs: a minimal
    config dict, the fitted encoder, the fitted scaler, and the raw
    feature column order (mirroring X_train.columns in the real pipeline).
    """
    training_df = pd.DataFrame({
        'tenure': [1, 24, 60, 12],
        'MonthlyCharges': [70.0, 50.0, 90.0, 65.0],
        'TotalCharges': [70.0, 1200.0, 5400.0, 780.0],
        'gender': [1, 0, 1, 0],              # already binary-encoded
        'Contract': [0, 1, 2, 0],            # already ordinal-encoded
        'InternetService': ['Fiber optic', 'DSL', 'No', 'DSL'],
    })

    nominal_cols = ['InternetService']
    feature_columns = training_df.columns.tolist()

    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    encoder.fit(training_df[nominal_cols])

    # Scaler is fit on the *encoded* data, matching train.py's real order
    # of operations (encode nominal columns first, then scale everything).
    encoded = encoder.transform(training_df[nominal_cols])
    encoded_cols = encoder.get_feature_names_out(nominal_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
    full_encoded = pd.concat(
        [training_df.drop(columns=nominal_cols), encoded_df], axis=1
    )
    scaler = StandardScaler()
    scaler.fit(full_encoded)

    config = {
        'features': {
            'binary_cols': [],
            'ordinal_mappings': {},
            'nominal_cols': nominal_cols,
        }
    }

    return config, encoder, scaler, feature_columns


def test_preprocess_customer_data_is_column_order_independent(fitted_pipeline_pieces):
    """
    THE key regression test for the column-order bug we found and fixed.

    Builds the same customer as two dicts - one with keys in the
    "correct" (training) order, one with keys deliberately scrambled -
    and confirms preprocess_customer_data() produces the *identical*
    scaled output for both.
    """
    config, encoder, scaler, feature_columns = fitted_pipeline_pieces

    ordered_customer = {
        'tenure': 5,
        'MonthlyCharges': 80.0,
        'TotalCharges': 400.0,
        'gender': 1,
        'Contract': 0,
        'InternetService': 'Fiber optic',
    }

    scrambled_customer = {
        'InternetService': 'Fiber optic',
        'Contract': 0,
        'gender': 1,
        'TotalCharges': 400.0,
        'tenure': 5,
        'MonthlyCharges': 80.0,
    }

    ordered_df = pd.DataFrame([ordered_customer])
    scrambled_df = pd.DataFrame([scrambled_customer])

    X_scaled_ordered = preprocess_customer_data(
        ordered_df, config, encoder, scaler, feature_columns
    )
    X_scaled_scrambled = preprocess_customer_data(
        scrambled_df, config, encoder, scaler, feature_columns
    )

    np.testing.assert_array_equal(X_scaled_ordered, X_scaled_scrambled)


def test_preprocess_customer_data_raises_on_missing_field(fitted_pipeline_pieces):
    """
    If a caller forgets a required field, preprocess_customer_data()
    should fail loudly with a clear ValueError - not silently reindex
    it into a NaN column and let that NaN flow into the model unnoticed.
    """
    config, encoder, scaler, feature_columns = fitted_pipeline_pieces

    incomplete_customer = {
        'tenure': 5,
        'MonthlyCharges': 80.0,
        'TotalCharges': 400.0,
        'gender': 1,
        'Contract': 0,
        # InternetService missing on purpose
    }
    incomplete_df = pd.DataFrame([incomplete_customer])

    with pytest.raises(ValueError, match="InternetService"):
        preprocess_customer_data(
            incomplete_df, config, encoder, scaler, feature_columns
        )