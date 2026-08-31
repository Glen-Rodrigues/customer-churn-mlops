"""
monitoring.py

Phase 7: Data drift detection.

Compares the training data (reference) against new/incoming data
(current) to detect when real-world data starts looking different
from what the model was trained on. This is the standard first line
of defense in production ML monitoring: you often don't get "did this
customer actually churn" labels for weeks, but you CAN check every day
whether the shape of incoming data has changed - often the earliest
hint something is off before accuracy visibly drops.

Uses Evidently's DataDriftPreset, which runs one statistical test per
column (Kolmogorov-Smirnov for numeric columns, chi-square/Z-test for
categorical columns) comparing the reference distribution to the
current one, and flags any column whose p-value falls below 0.05.
"""

import os
import pandas as pd
import numpy as np
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset
from data_preprocessing import load_config


def load_reference_data(config):
    """
    Load train.csv as the reference distribution - i.e. "what did the
    world look like when the model was trained". Drops the target
    column (Churn) since drift monitoring here is about whether INPUT
    features have shifted, not the label itself.
    """
    train_path = os.path.join(config['data']['processed_dir'], 'train.csv')
    df = pd.read_csv(train_path)
    df = df.drop(columns=[config['data']['target_column']])
    return df


def load_current_data(config, filename='test.csv'):
    """
    Load a "current" dataset to compare against the reference.
    Defaults to test.csv, standing in for "new data arriving after
    deployment" since there's no real production traffic yet.
    """
    path = os.path.join(config['data']['processed_dir'], filename)
    df = pd.read_csv(path)
    if config['data']['target_column'] in df.columns:
        df = df.drop(columns=[config['data']['target_column']])
    return df


def generate_drift_report(reference_df, current_df):
    """
    Run Evidently's DataDriftPreset comparing reference_df vs current_df.
    Returns the raw result object (an Evidently Snapshot), which can
    then be saved as HTML or summarized into a plain table.
    """
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        reference_data=Dataset.from_pandas(reference_df),
        current_data=Dataset.from_pandas(current_df),
    )
    return result


def save_report_html(result, output_path):
    """Save the full interactive Evidently report as a standalone HTML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save_html(output_path)
    return output_path


def summarize_drift(result):
    """
    Pull the "at a glance" pieces out of the Evidently result: how many
    columns drifted overall, and a per-column table (test used, value,
    drifted yes/no).

    Evidently picks the statistical test per column automatically based
    on data size and type - small samples get p-value tests (K-S,
    chi-square, Z-test), larger samples get distance-based tests
    (Jensen-Shannon distance, Wasserstein distance), since p-value tests
    get overly sensitive to trivial differences once you have thousands
    of rows. Critically, the two types are interpreted in OPPOSITE
    directions:
      - p-value tests: drifted when value is BELOW the threshold
        (unlikely the two samples come from the same distribution)
      - distance tests: drifted when value is ABOVE the threshold
        (the two distributions are far apart)
    so the comparison direction has to switch based on which test ran.
    """
    metrics = result.dict()['metrics']

    overall = next(m for m in metrics if m['metric_name'].startswith('DriftedColumnsCount'))
    n_drifted = int(overall['value']['count'])
    share_drifted = overall['value']['share']

    rows = []
    for m in metrics:
        if m['metric_name'].startswith('ValueDrift'):
            method = m['config']['method']
            threshold = m['config']['threshold']
            value = m['value']

            is_p_value_test = 'p_value' in method
            drifted = (value < threshold) if is_p_value_test else (value > threshold)

            rows.append({
                'column': m['config']['column'],
                'test': method,
                'value': round(value, 4),
                'drifted': drifted,
            })

    summary_df = pd.DataFrame(rows).sort_values('value', ascending=False).reset_index(drop=True)
    return n_drifted, share_drifted, summary_df


def apply_price_increase(df, pct_increase=0.15):
    """
    Simulate a price increase: bump MonthlyCharges by pct_increase
    (default 15%) and scale TotalCharges by the same factor, so the two
    stay proportionally consistent with each other.

    Simplification worth being upfront about: TotalCharges is what a
    customer has already been billed to date, so a price increase
    wouldn't retroactively change their past total. Scaling it here is
    a deliberate simplification standing in for "these are newer
    customers who signed up after the price increase took effect" -
    not a claim that existing customers' historical bills changed.
    """
    df = df.copy()
    df['MonthlyCharges'] = df['MonthlyCharges'] * (1 + pct_increase)
    df['TotalCharges'] = df['TotalCharges'] * (1 + pct_increase)
    return df


def apply_segment_shift(df, shift_fraction=0.15, random_state=42):
    """
    Simulate a shift toward a riskier customer mix: randomly picks
    shift_fraction (default 15%) of rows and pushes them toward the
    highest-churn-risk categories found in Phase 1 EDA - Contract
    becomes Month-to-month, InternetService becomes Fiber optic,
    PaymentMethod becomes Electronic check. Standing in for something
    like a marketing push that brought in a differently-shaped batch
    of customers, rather than a pricing change (that's Piece 1).
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    n_rows = len(df)
    n_shift = int(n_rows * shift_fraction)
    shift_idx = rng.choice(df.index, size=n_shift, replace=False)

    df.loc[shift_idx, 'Contract'] = 0                    # 0 = Month-to-month (ordinal encoding)
    df.loc[shift_idx, 'InternetService'] = 'Fiber optic'
    df.loc[shift_idx, 'PaymentMethod'] = 'Electronic check'

    return df


def main():
    """
    CLI entry point: generates two drift reports against the same
    reference (train.csv):
      1. Baseline - test.csv, unmodified. Expected: little to no drift,
         since it's already confirmed clean (Step 1).
      2. Simulated drift - test.csv run through apply_price_increase()
         and apply_segment_shift(), standing in for a "bad quarter"
         (price hike + a riskier new customer mix) since there's no
         real production traffic to check drift against yet.
    """
    config = load_config()
    reference_df = load_reference_data(config)
    reports_dir = config['monitoring']['reports_dir']

    # 1. Baseline (unchanged from Step 1)
    baseline_current_df = load_current_data(config, filename='test.csv')
    baseline_result = generate_drift_report(reference_df, baseline_current_df)
    save_report_html(baseline_result, os.path.join(reports_dir, 'drift_report_baseline.html'))
    n_drifted, share_drifted, summary_df = summarize_drift(baseline_result)
    print("=== Baseline (no drift expected) ===")
    print(f"Drifted columns: {n_drifted} / {len(summary_df)} ({share_drifted:.0%})")
    print(summary_df.to_string(index=False))

    # 2. Simulated drift
    drifted_current_df = load_current_data(config, filename='test.csv')
    drifted_current_df = apply_price_increase(drifted_current_df)
    drifted_current_df = apply_segment_shift(drifted_current_df, shift_fraction=0.35)
    drifted_result = generate_drift_report(reference_df, drifted_current_df)
    save_report_html(drifted_result, os.path.join(reports_dir, 'drift_report_simulated.html'))
    n_drifted2, share_drifted2, summary_df2 = summarize_drift(drifted_result)
    print("\n=== Simulated drift (price increase + segment shift) ===")
    print(f"Drifted columns: {n_drifted2} / {len(summary_df2)} ({share_drifted2:.0%})")
    print(summary_df2.to_string(index=False))


if __name__ == "__main__":
    main()