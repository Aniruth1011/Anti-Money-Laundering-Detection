from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def save_evaluation_plots(y_true: np.ndarray, scores: np.ndarray, output_dir: str | Path):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    threshold = float(np.quantile(scores, 0.98))
    y_pred = (scores >= threshold).astype(int)
    paths = {
        "confusion_matrix": output / "confusion_matrix.png",
        "roc_curve": output / "roc_curve.png",
        "pr_curve": output / "pr_curve.png",
        "score_distribution": output / "score_distribution.png",
    }
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
    plt.savefig(paths["confusion_matrix"], bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_predictions(y_true, scores)
    plt.savefig(paths["roc_curve"], bbox_inches="tight")
    plt.close()
    PrecisionRecallDisplay.from_predictions(y_true, scores)
    plt.savefig(paths["pr_curve"], bbox_inches="tight")
    plt.close()
    plt.hist(scores[y_true == 0], bins=50, alpha=0.7, label="legitimate")
    plt.hist(scores[y_true == 1], bins=50, alpha=0.7, label="fraud")
    plt.legend()
    plt.savefig(paths["score_distribution"], bbox_inches="tight")
    plt.close()
    return paths
