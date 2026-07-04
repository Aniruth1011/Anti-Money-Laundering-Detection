from __future__ import annotations

import time
from dataclasses import dataclass

import mlflow.pyfunc
import pandas as pd

from aml_fraud.streaming.replay import ReplayEngine


@dataclass
class StreamingPrediction:
    transaction_id: int
    score: float
    prediction: int
    latency_ms: float


class FraudStreamingPipeline:
    def __init__(self, model_uri: str, threshold: float = 0.5):
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.threshold = threshold

    def predict_event(self, payload: dict, transaction_id: int):
        start = time.perf_counter()
        score = float(self.model.predict(pd.DataFrame([payload]))[0])
        latency = (time.perf_counter() - start) * 1000
        return StreamingPrediction(transaction_id, score, int(score >= self.threshold), latency)

    def replay(self, frame: pd.DataFrame, speed: float = 1.0):
        engine = ReplayEngine(frame, speed=speed)
        return [self.predict_event(event.payload, event.transaction_id) for event in engine.events()]
