from __future__ import annotations

import os
import time
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(min_length=1)

class ModelServer:
    def __init__(self):
        model_uri = os.getenv("MODEL_URI", "models:/aml_fraud_best_model/Production")
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, rows: list[dict[str, Any]]):
        start = time.perf_counter()
        scores = self.model.predict(pd.DataFrame(rows))
        latency = (time.perf_counter() - start) * 1000 / len(rows)
        return {"scores": [float(score) for score in scores], "latency_ms": latency}


app = FastAPI(title="AML Fraud Detection API")
server: ModelServer | None = None


@app.on_event("startup")
def startup():
    global server
    server = ModelServer()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"model_uri": server.model_uri if server else "not-loaded"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if server is None:
        startup()
    return server.predict(request.transactions)
