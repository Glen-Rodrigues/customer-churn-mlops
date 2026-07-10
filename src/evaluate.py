"""
evaluate.py

Loads the champion model (selected in Phase 3, identified by MLflow run_id
in config.yaml) and runs deeper evaluation on it: confusion matrix, ROC
curve, and SHAP feature importance.

Design principle (matches train.py): small, focused, testable functions,
with a main() orchestrator at the bottom that chains them together.

Why load instead of retrain: the champion was picked by comparing 8 MLflow
runs on specific multi-metric criteria (see project_status.md). Loading the
exact logged artifact guarantees this evaluation reflects that exact model
- retraining fresh here could drift slightly (different random state
handling internally, library version differences, etc.) and silently
evaluate a DIFFERENT model than the one actually selected.
"""

import mlflow
import mlflow.lightgbm
from data_preprocessing import load_config
from train import load_processed_data, scale_features, evaluate_model
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import shap
import numpy as np
import pandas as pd


def load_champion_model(config):
    """
    Load the champion LightGBM model directly from its MLflow run.

    Reads config['mlflow']['champion_run_id'] (set once in Phase 3 after
    picking the champion) and fetches that run's saved model artifact.
    Returns a real LGBMClassifier object, ready to call .predict() /
    .predict_proba() on, exactly as if we'd just trained it ourselves.
    """
    run_id = config['mlflow']['champion_run_id']
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)
    return model


def sanity_check_champion(model, X_test_scaled, y_test):
    """
    Re-run train.py's evaluate_model() on the loaded champion and print
    the result, to confirm this is really the model documented as the
    Phase 3 champion (F1 0.648, recall 0.818, roc_auc 0.857) before
    building anything new on top of it. Reuses evaluate_model() rather
    than duplicating metric-computation logic - single source of truth
    for "how do we score a model" stays in train.py.
    """
    metrics = evaluate_model(model, X_test_scaled, y_test)
    print("Sanity check - champion model metrics on test set:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    return metrics

def get_confusion_matrix(model, X_test_scaled, y_test):
    """
    Compute the confusion matrix for the champion model on the test set.

    Returns a 2x2 array: [[TN, FP], [FN, TP]]. TN/TP are correct
    predictions; FP is a false alarm (predicted churn, customer stayed);
    FN is a missed churner (predicted stay, customer actually churned) -
    the costliest mistake per the business-cost-asymmetry principle
    already documented in project_status.md.
    """
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    return cm


def plot_confusion_matrix(cm, save_path):
    """
    Render the confusion matrix as a labeled heatmap and save it to disk.

    annot=True prints the actual counts inside each cell (not just
    color); fmt='d' keeps them as whole numbers (counts, not decimals).
    Labeled axes matter here since an unlabeled 2x2 grid of numbers is
    meaningless without knowing which axis is "actual" vs "predicted".
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Champion Model (LightGBM Tuned)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(model, X_test_scaled, y_test, save_path):
    """
    Plot the ROC curve for the champion model and save it to disk.

    roc_curve() sweeps through many possible decision thresholds and
    returns the False Positive Rate (fpr) and True Positive Rate (tpr)
    at each one. auc() computes the area under that curve - same number
    as evaluate_model()'s roc_auc metric, computed here directly from
    the curve so the plotted line and the printed AUC value are
    guaranteed consistent with each other.

    Uses predict_proba (not predict) because the whole point of this
    plot is to see performance BEFORE collapsing probabilities down to
    a single 0.5-threshold decision.
    """
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Champion Model (LightGBM Tuned)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

    return roc_auc

def compute_shap_values(model, X_test_scaled, feature_names):
    """
    Compute SHAP values for the champion model on the test set.

    TreeExplainer is used specifically because LightGBM is a tree-based
    model - it computes exact SHAP values efficiently for tree ensembles,
    rather than the slower/approximate general-purpose Explainer meant
    for arbitrary model types (e.g. neural nets).

    X_test_scaled is a plain NumPy array (StandardScaler strips column
    names - the same cosmetic issue already logged in project_status.md),
    so feature_names is passed in separately to label the plot correctly.
    Wrapped back into a DataFrame here specifically for SHAP's plotting
    functions, which expect column names to label the chart.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Newer SHAP versions return a list [class_0_values, class_1_values]
    # for binary classifiers instead of one array. We only want the
    # "pushes toward Churn=1" values - that's index 1, matching how
    # encode_target_column() defined the target (Churn Yes -> 1).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    return shap_values, X_test_df


def plot_shap_summary(shap_values, X_test_df, save_path):
    """
    Plot and save a SHAP summary plot: ranks features by overall
    importance (mean absolute SHAP value), and shows whether high/low
    values of each feature push predictions toward or away from churn.
    """
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")


def analyze_false_negatives(model, X_test, X_test_scaled, y_test):
    """
    Compare missed churners (false negatives) against correctly-caught
    churners (true positives), feature by feature, using unscaled X_test
    so values are human-readable (real tenure months, real dollar
    amounts) rather than standardized scores.

    Purpose: the confusion matrix told us HOW MANY churners were missed;
    this tells us WHAT KIND of churner the model tends to miss - a
    concrete, quotable finding rather than just an aggregate recall %.
    """
    y_pred = model.predict(X_test_scaled)
    y_test_reset = y_test.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)

    fn_mask = (y_test_reset == 1) & (y_pred == 0)
    tp_mask = (y_test_reset == 1) & (y_pred == 1)

    print(f"\nMissed churners (FN): {fn_mask.sum()} | Caught churners (TP): {tp_mask.sum()}")

    comparison = pd.DataFrame({
        "missed_churners_avg": X_test_reset[fn_mask].mean(),
        "caught_churners_avg": X_test_reset[tp_mask].mean(),
    })
    comparison["difference"] = comparison["missed_churners_avg"] - comparison["caught_churners_avg"]
    comparison = comparison.reindex(comparison["difference"].abs().sort_values(ascending=False).index)

    print("\nBiggest differences between missed vs caught churners:")
    print(comparison.head(10))
    return comparison


def compare_feature_importance(model, shap_values, feature_names):
    """
    Cross-check SHAP's importance ranking against LightGBM's own
    built-in feature_importances_ (split-count based). They measure
    different things (usage frequency vs actual prediction impact) and
    can legitimately disagree - showing both is more credible than
    presenting SHAP as the only lens on "what matters".
    """
    lgbm_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)

    print("\nTop 10 - LightGBM built-in importance (split count):")
    print(lgbm_importance.head(10))
    print("\nTop 10 - SHAP mean |impact| on prediction:")
    print(shap_importance.head(10))


def main():
    """
    Orchestrates the full Phase 4 evaluation pipeline: load champion
    model, sanity-check it against Phase 3 metrics, then run confusion
    matrix, ROC curve, SHAP, and false-negative analysis - all against
    the same loaded model and test set for consistency.
    """
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    model = load_champion_model(config)
    print("Champion model loaded successfully:")
    print(model)

    X_train, X_test, y_train, y_test = load_processed_data(
        config['data']['processed_dir'],
        config['data']['target_column']
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    sanity_check_champion(model, X_test_scaled, y_test)

    cm = get_confusion_matrix(model, X_test_scaled, y_test)
    print("\nConfusion matrix:")
    print(cm)
    plot_confusion_matrix(cm, "artifacts/confusion_matrix.png")

    roc_auc = plot_roc_curve(model, X_test_scaled, y_test, "artifacts/roc_curve.png")
    print(f"ROC-AUC: {roc_auc:.4f}")

    shap_values, X_test_df = compute_shap_values(model, X_test_scaled, X_train.columns)
    plot_shap_summary(shap_values, X_test_df, "artifacts/shap_summary.png")

    analyze_false_negatives(model, X_test, X_test_scaled, y_test)
    compare_feature_importance(model, shap_values, X_train.columns)


if __name__ == "__main__":
    main()