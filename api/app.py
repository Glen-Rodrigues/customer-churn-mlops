"""
app.py

FastAPI web server that exposes the champion churn model as a REST API.
Wraps predict.py's existing predict_single() function - this file adds
NO new prediction logic, it just gives HTTP access to logic that
already exists and is already tested.
"""

import sys
import os

"""
api/app.py lives in api/, but load_artifacts, predict_single, etc. live
in src/. Python only auto-adds a script's OWN folder to the import
path, not sibling folders - so without this, "from predict import ..."
below would fail with ModuleNotFoundError.
This is the exact same problem (and same fix) as tests/conftest.py
solving it for pytest - just done here for uvicorn instead.
"""
SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_PATH))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal


"""
This mirrors the exact 19 raw feature columns your model was trained
on (see config.yaml's binary_cols/nominal_cols/ordinal_mappings, plus
the 4 already-numeric passthrough columns: SeniorCitizen, tenure,
MonthlyCharges, TotalCharges). This is the SAME shape of dict that
predict_single() in predict.py already expects.

Categorical fields use Literal instead of plain str - this is the
"strict validation" choice: FastAPI will reject a request with e.g.
InternetService="Fiber Optic" (wrong case) or a typo, with a clear
422 error, BEFORE it ever reaches the model. Without this, a typo
would silently get treated as an "unknown category" by the encoder
and produce a confidently wrong prediction with no error at all -
same failure class as the column-order bug from Phase 5.
"""
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float
    TotalCharges: float


"""
This is the shape of what /predict sends BACK. Just as strict as the
request - callers know exactly what fields to expect, no guessing.
"""
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str


from contextlib import asynccontextmanager
from predict import load_artifacts, predict_single
from data_preprocessing import load_config

"""
Build an absolute path to config.yaml based on THIS file's location,
not on whatever folder someone happens to run uvicorn from. If we
used a relative path like "configs/config.yaml", it would only work
when launched from the repo root - and would silently break inside
Docker if the container's WORKDIR is set up differently. This same
"don't trust the working directory" instinct is why predict.py loads
things via config paths in the first place.
"""
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

"""
Holds the loaded model/encoder/scaler/feature_columns after startup,
so the /predict endpoint can reach them without reloading anything.
"""
ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    STARTUP (runs once, before the server accepts any requests):
    load the champion model + encoder + scaler + column order a
    single time and keep them in memory. This matters a lot -
    loading the MLflow model and joblib artifacts from disk takes
    real time (hundreds of ms to seconds). If we loaded them INSIDE
    the /predict function instead, every single request would pay
    that cost again, making the API needlessly slow and hammering
    disk/MLflow for no reason - the model itself never changes
    between requests, so there's nothing to gain by reloading it.
    """
    config = load_config(CONFIG_PATH)
    model, encoder, scaler, feature_columns = load_artifacts(config)
    ml_artifacts["model"] = model
    ml_artifacts["encoder"] = encoder
    ml_artifacts["scaler"] = scaler
    ml_artifacts["feature_columns"] = feature_columns
    ml_artifacts["config"] = config

    yield  # the API runs and serves requests here

    """
    SHUTDOWN (runs once, when the server stops): nothing to release
    here (no open DB connections or file handles), but this is where
    that kind of cleanup would go if we ever needed it.
    """
    ml_artifacts.clear()


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


"""
Simple "is this thing alive" check - no model logic involved. Useful
for Docker healthchecks later, and for you to quickly confirm the
server started correctly before testing the real endpoint.
"""
@app.get("/health")
def health_check():
    return {"status": "ok"}


"""
The actual prediction endpoint. FastAPI wires this up automatically:
- customer: CustomerData means FastAPI parses+validates the incoming
  JSON body against the schema from Part 2 BEFORE this function body
  even runs. If validation fails, the caller gets a 422 error and
  predict_single() is never called.
- response_model=PredictionResponse means FastAPI also checks that
  whatever we return matches that shape, and documents it in the
  auto-generated API docs.
"""
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """
    .model_dump() turns the validated Pydantic object back into a
    plain dict - customer_dict - which is exactly the input shape
    predict_single() (from predict.py) already expects and is
    already tested against. No new prediction logic lives here.
    """
    customer_dict = customer.model_dump()

    result = predict_single(
        customer_dict,
        ml_artifacts["model"],
        ml_artifacts["encoder"],
        ml_artifacts["scaler"],
        ml_artifacts["feature_columns"],
        ml_artifacts["config"],
    )
    return result