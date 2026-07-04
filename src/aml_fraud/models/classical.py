from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

try:
    from xgboost import XGBClassifier
except ImportError: 
    XGBClassifier = None

@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    semi_supervised: bool = False


def build_classical_models(random_state: int = 42, contamination: float = 0.02):
    models = [
        ModelSpec("isolation_forest", IsolationForest(contamination=contamination, random_state=random_state), True),
        ModelSpec("local_outlier_factor", LocalOutlierFactor(contamination=contamination, novelty=True), True),
        ModelSpec("one_class_svm", OneClassSVM(kernel="rbf", nu=contamination), True),
        ModelSpec("random_forest", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=random_state)),
    ]
    if XGBClassifier is not None:
        models.append(
            ModelSpec("xgboost",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    eval_metric="logloss",
                    random_state=random_state,
                ),
            )
        )
    return models


def fit_model(spec: ModelSpec, transformer, x_train, y_train):
    if spec.semi_supervised:
        x_fit = x_train[y_train == 0]
        y_fit = None
    else:
        x_fit = x_train
        y_fit = y_train
    pipeline = Pipeline([("features", transformer), ("model", spec.estimator)])
    if y_fit is None:
        pipeline.fit(x_fit)
    else:
        pipeline.fit(x_fit, y_fit)
    return pipeline
