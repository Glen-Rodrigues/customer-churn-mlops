"""
monitoring/dashboard.py

Phase 7: Streamlit dashboard for interactively viewing data drift.

Deliberately does NOT reimplement any drift logic - it imports and
reuses the functions already built and verified in src/monitoring.py
(load_config, load_reference_data, load_current_data,
generate_drift_report, summarize_drift). This file is purely a UI
layer on top of that, same "translation layer, no new logic"
principle used for api/app.py in Phase 6.
"""

import sys
import os

# src/ is a sibling folder to monitoring/, not auto-discovered by Python.
# Same fix already used in api/app.py and tests/conftest.py.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from monitoring import (
    load_config,
    load_reference_data,
    load_current_data,
    generate_drift_report,
    summarize_drift,
    apply_price_increase,
    apply_segment_shift,
)

# set_page_config must be the FIRST Streamlit command in the script,
# or Streamlit throws an error. Sets the browser tab title/layout.
st.set_page_config(page_title="Churn Model - Drift Monitor", layout="wide")


@st.cache_data
def get_reference_data():
    """
    Load train.csv (the reference distribution) once and cache it.
    Cached because it never changes between dropdown selections -
    reloading it on every rerun would be pure wasted work.
    """
    config = load_config()
    return load_reference_data(config), config


@st.cache_data
def get_current_data(source, config):
    """
    Load the "current" dataset to compare against reference, based on
    the dropdown selection. Cached per distinct `source` value, so
    switching between baseline/simulated re-uses cached results if
    you switch back.

    NOTE: config is passed in as an argument (not read from a global)
    because @st.cache_data needs all its inputs to be visible in the
    function signature to know when to invalidate the cache.
    """
    if source == "Baseline (test.csv, unmodified)":
        return load_current_data(config, filename='test.csv')
    else:  # "Simulated drift"
        df = load_current_data(config, filename='test.csv')
        df = apply_price_increase(df)
        df = apply_segment_shift(df, shift_fraction=0.35)
        return df


@st.cache_data
def get_drift_summary(_reference_df, _current_df, cache_key):
    """
    Run the actual drift comparison and summarize it into a table.
    Reuses generate_drift_report() + summarize_drift() from
    monitoring.py unchanged - no drift-detection logic lives here.

    The leading underscore on _reference_df/_current_df tells
    st.cache_data NOT to try hashing these DataFrames directly (which
    can be slow/unreliable) - instead we pass a separate small
    `cache_key` string (the dropdown value) that IS cheap to hash, and
    Streamlit uses that to decide whether to reuse a cached result.
    """
    result = generate_drift_report(_reference_df, _current_df)
    n_drifted, share_drifted, summary_df = summarize_drift(result)
    return n_drifted, share_drifted, summary_df


# --- Page layout starts here ---

st.title("Customer Churn Model - Data Drift Monitor")
st.caption(
    "Compares incoming customer data against the training data "
    "(train.csv) the champion model was trained on."
)

reference_df, config = get_reference_data()

# Sidebar dropdown for picking the data source to compare against reference.
source = st.sidebar.selectbox(
    "Current data source",
    options=["Baseline (test.csv, unmodified)", "Simulated drift (price increase + segment shift)"],
)

current_df = get_current_data(source, config)
n_drifted, share_drifted, summary_df = get_drift_summary(reference_df, current_df, source)

# st.metric renders a big, glanceable number card - good for the
# headline "how bad is it" stat.
col1, col2 = st.columns(2)
col1.metric("Columns drifted", f"{n_drifted} / {len(summary_df)}")
col2.metric("Share drifted", f"{share_drifted:.0%}")

st.subheader("Per-column drift detail")
st.dataframe(summary_df, use_container_width=True)