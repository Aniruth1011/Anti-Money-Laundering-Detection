from __future__ import annotations

import argparse
import time

import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split

from aml_fraud.evaluation.metrics import evaluate_scores, timed_scores
from aml_fraud.features.builder import FeatureBuilder
from aml_fraud.models.classical import build_classical_models, fit_model
from aml_fraud.models.mlflow_utils import configure_mlflow
from aml_fraud.preprocessing.amlsim_loader import build_training_tables
from aml_fraud.preprocessing.pipeline import AMLPreprocessor
from aml_fraud.utils.config import load_config

def train(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    data_cfg = config["data"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]
    configure_mlflow(mlflow_cfg["tracking_uri"], train_cfg["experiment_name"])

    output_dir = Path(data_cfg.get("amlsim_output_dir", ""))
    if output_dir.exists():
        transactions, accounts = build_training_tables(output_dir)
        frame = AMLPreprocessor().transform(transactions, accounts)
    else:
        frame = AMLPreprocessor().load(data_cfg["transactions_path"], data_cfg.get("accounts_path"))
    feature_builder = FeatureBuilder()
    features = feature_builder.build_frame(frame)
    x, y = feature_builder.split_xy(features)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=train_cfg["test_size"],random_state=train_cfg["random_state"],stratify=y if len(set(y)) > 1 else None,)

    for spec in build_classical_models(train_cfg["random_state"], train_cfg["contamination"]):
        transformer = feature_builder.make_transformer(x_train)
        start = time.perf_counter()
        model = fit_model(spec, transformer, x_train, y_train)
        training_time = time.perf_counter() - start
        scores, inference_time = timed_scores(model, x_test)
        result = evaluate_scores(y_test, scores, elapsed=inference_time)
        params = {"model": spec.name, "preprocessing_version": config["features"]["preprocessing_version"], "feature_version": config["features"]["feature_version"],}
        metrics = result.__dict__ | {"training_time": training_time}
        with mlflow.start_run(run_name=spec.name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model", registered_model_name=mlflow_cfg["registry_model_name"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)

if __name__ == "__main__":
    main() 
