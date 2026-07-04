from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    inference_latency_ms: float


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None, elapsed: float = 0.0):
    if threshold is None:
        threshold = float(np.quantile(scores, 0.98))
    y_pred = (scores >= threshold).astype(int)
    latency = elapsed * 1000 / max(len(scores), 1)
    return EvaluationResult(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=_safe_auc(roc_auc_score, y_true, scores),
        pr_auc=_safe_auc(average_precision_score, y_true, scores),
        inference_latency_ms=latency,
    )


def timed_scores(model, x):
    start = time.perf_counter()
    if hasattr(model, "decision_function"):
        scores = -model.decision_function(x)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    else:
        scores = model.predict(x)
    return np.asarray(scores), time.perf_counter() - start


def _safe_auc(fn, y_true: np.ndarray, scores: np.ndarray):
    return float(fn(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0
