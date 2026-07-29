# Base image - official Python 3.13 (matches CI and local dev exactly,
# see .github/workflows/tests.yml). "slim" variant strips unnecessary
# OS packages to keep the image smaller/faster without losing anything
# our stack (LightGBM, XGBoost, FastAPI, etc.) needs.
FROM python:3.13-slim

# All following commands run as if we'd cd'd into /app inside the
# container. This is what makes relative paths (config.yaml paths,
# api/app.py's __file__-based CONFIG_PATH) resolve consistently
# regardless of where/how the container gets launched.
WORKDIR /app

# LightGBM's compiled binary depends on libgomp (OpenMP runtime) at
# import time, which the slim base image doesn't include by default.
# Installed here as a system package (not pip) since it's a native
# shared library, not a Python package. --no-install-recommends and
# clearing apt lists after keep the image from bloating with docs/
# extras we don't need.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy ONLY the dependency manifest first, install right after, and
# copy the actual code later (below). Docker caches each layer based
# on whether its input changed - since requirements.txt changes far
# less often than src/api code, this ordering means most rebuilds
# skip re-downloading/reinstalling every package and jump straight to
# the fast "copy code" layers.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Explicit COPY per folder (not one COPY . .) for two reasons:
# 1) keeps caching granular - editing src/ doesn't force redoing the
#    configs/ or mlruns/ layers, and vice versa.
# 2) avoids accidentally shipping .git/, notebooks/, tests/, __pycache__/
#    etc. into the image - only what api/app.py actually needs at
#    runtime gets copied in.
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# artifacts/ (encoder, scaler, feature_columns, and the exported
# champion_model/ folder) is gitignored (see project_status.md Phase 6
# design decision) - COPY pulls it from local disk at build time
# regardless of .gitignore, since git and docker build don't share
# that rule. Deliberate tradeoff: this Dockerfile can only build on a
# machine that already has artifacts/ populated (i.e. one that's
# already run train.py AND export_champion_model.py locally) - a DVC
# remote + `dvc pull` step is the more "correct" long-term fix, parked
# for later (see project_status.md).
#
# mlruns/ and mlflow.db are deliberately NOT copied in anymore. Early
# in Phase 6 the API loaded the champion model via MLflow's tracking
# store (runs:/{run_id}/model), which needed both files at runtime -
# that's why they were here originally. Since the Option B fix
# (export_champion_model.py + load_champion_model_from_export() in
# predict.py), the Docker/API path loads the model as plain files from
# artifacts/champion_model/ and never touches mlruns/ or mlflow.db at
# all. Copying them in was harmless-but-wasted image size and build
# time - evaluate.py still uses them, but evaluate.py only ever runs
# locally, never inside this container.
COPY artifacts/ ./artifacts/

# Documents which port the container listens on. Does NOT actually
# publish the port to the host machine by itself - that happens at
# `docker run -p ...` time. This is just a label/contract, not
# enforcement.
EXPOSE 8000

# Starts uvicorn every time a container runs from this image (unlike
# COPY/RUN above, which only happen once, at build time).
# - api.app:app -> look in the api package, app.py file, find the
#   `app = FastAPI(...)` variable - only resolvable because api/ was
#   copied to /app/api/ and WORKDIR is /app.
# - --host 0.0.0.0 is required, not optional: without it uvicorn
#   defaults to 127.0.0.1 (container-internal only), and requests
#   from outside the container (our browser/Postman) would just time
#   out with no obvious error.
# - --port 8000 matches EXPOSE above and what we'll publish via
#   `docker run -p 8000:8000`.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]