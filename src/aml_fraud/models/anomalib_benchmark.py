from __future__ import annotations


def explain_anomalib_scope():
    return (
        "Anomalib is primarily designed around visual anomaly detection. "
        "GANomaly can be used as a conceptual benchmark, but image-first models "
        "such as PatchCore are not appropriate for raw tabular AML transactions "
        "without an artificial image representation, so this project keeps them "
        "documented instead of forcing a misleading implementation."
    )
