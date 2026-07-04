from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


def configure_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_run(model_name: str, params: dict[str, Any], metrics: dict[str, float], model: Any, registry_name: str | None = None):
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model is not None:
            mlflow.sklearn.log_model(model, "model", registered_model_name=registry_name)
        return run.info.run_id


def ensure_artifact_dir(path: str | Path = "artifacts"):
    artifact_dir = Path(path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir
