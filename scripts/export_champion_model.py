"""
export_champion_model.py

One-time (rerun only when the champion changes) utility that re-saves
the champion model from MLflow's tracking store to a plain, portable
local folder.

Why this exists: mlflow.db records the ABSOLUTE local filesystem path
to each run's artifacts at the moment train.py was run (e.g.
"W:/Projects/customer-churn-mlops/mlruns/..."). That's fine on the
machine that trained the model, but breaks the moment the model needs
to load somewhere else - like inside a Docker container, where that
Windows path doesn't exist. Re-saving the model as a self-contained
folder (via mlflow.lightgbm.save_model) removes that dependency
entirely - predict.py can load it as plain files, no tracking store
or run_id lookup required.

NOTE: this script used to also export feature_columns to a JSON file.
That's been removed - train.py now saves feature_columns.joblib
automatically on every training run (same principle as the encoder/
scaler), so a second, separate export step for it was redundant and
risked overwriting the correct joblib file with a stale JSON one if
this script were ever rerun without also touching feature_columns.
There is one orphaned leftover from the old version of this script:
artifacts/feature_columns.json. It's unused now (predict.py loads
feature_columns.joblib) - safe to delete.

Run this locally (not in Docker) whenever config.yaml's
champion_run_id changes - e.g. after retraining and picking a new
champion. Must run BEFORE `docker build`, since the Dockerfile's
COPY artifacts/ ./artifacts/ line picks up whatever's in this folder
at build time.
"""

import shutil
import os
import mlflow
import mlflow.lightgbm
from data_preprocessing import load_config


def export_champion_model(config):
    """
    Load the champion model via its MLflow run_id (same as
    evaluate.py's load_champion_model), then save a plain, portable
    copy to config['artifacts']['champion_model_dir'].

    Uses mlflow.lightgbm.save_model() (not just copying files) so the
    output is a proper, self-describing MLflow model folder - same
    MLmodel/metadata structure, just anchored to a fixed local path
    instead of resolved through the tracking store's run_id lookup.

    mlflow.lightgbm.save_model() refuses to write into a non-empty
    folder (a safety guard against silently overwriting a model) - so
    since this script is meant to be rerun whenever the champion
    changes, we clear out any previous export first, making reruns
    safe without a manual delete step each time.
    """
    run_id = config['mlflow']['champion_run_id']
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)

    output_dir = config['artifacts']['champion_model_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    mlflow.lightgbm.save_model(model, output_dir)
    print(f"Champion model (run_id={run_id}) exported to: {output_dir}")


def main():
    """
    Entry point - loads config, points MLflow at the tracking store
    (needed here since we're loading FROM it via runs:/{run_id}/model),
    and exports the champion model.
    """
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    export_champion_model(config)


if __name__ == "__main__":
    main()