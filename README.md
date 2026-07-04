# IBM AMLSim Fraud Detection + MLOps

Production-oriented fraud detection project for IBM AMLSim-style transactions. It compares anomaly detection, supervised baselines, custom PyTorch models, graph learning, MLflow tracking/registry, FastAPI serving, and chronological replay simulation.

## What is included

- Reusable preprocessing with account metadata joins.
- Transaction, sender, receiver, account, and NetworkX graph features.
- Classical anomaly detection: Isolation Forest, Local Outlier Factor, One-Class SVM.
- Supervised baselines: Random Forest and XGBoost.
- Custom PyTorch models: Autoencoder, VAE, GANomaly, and HybridAnomalyNet.
- PyTorch Geometric model definitions: Graph Autoencoder, GraphSAGE, GCN encoder.
- MLflow tracking and model registry integration.
- FastAPI model serving from the MLflow registry.
- Streaming replay engine that replays real dataset rows chronologically.
- Docker and GitHub Actions for basic CI.

## Data

Generate AMLSim output under:

```text
data/AMLSim/outputs/sample/
```

The project uses these AMLSim files:

- `accounts.csv`
- `transactions.csv`
- `alert_accounts.csv`
- `alert_transactions.csv`
- `sar_accounts.csv`
- `tx_log.csv`

Normalize them into project-ready files with:

```bash
python scripts/prepare_amlsim_data.py --source data/AMLSim/outputs/sample --output data
```

The normalized transaction file includes:

- `timestamp`
- `sender_account`
- `receiver_account`
- `amount`
- `transaction_type`
- `is_fraud`
- alert/SAR flags
- sender/receiver tx-log balance fields
- simulation step

The normalized account file includes:

- `account_id`
- `account_type`
- `country`
- `initial_balance`
- SAR/account alert indicators

The preprocessor also accepts common aliases such as `orig_acct`, `dest_acct`, `tx_amount`, and `fraud`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Train classical and supervised models

```bash
python -m aml_fraud.models.train --config configs/default.yaml
```

Runs are logged to `mlruns/`, including parameters, metrics, and registered sklearn models.

## Serve a registered model

```bash
export MODEL_URI=models:/aml_fraud_best_model/Production
uvicorn aml_fraud.serving.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`

## Streaming replay

`aml_fraud.streaming.ReplayEngine` replays the historical dataset in timestamp order. The replay engine emits plain events so Kafka or another transport can be added later without changing feature or inference code.

## Notebooks

The notebooks are lightweight workflow guides:

- `01_eda.ipynb`: fraud distribution, amounts, types, account stats.
- `02_feature_engineering.ipynb`: engineered transaction, account, and graph features.
- `03_graph_analysis.ipynb`: graph inspection, centrality, fan-in, cycles.
- `04_model_comparison.ipynb`: metrics, latency, ROC and PR comparison.
- `05_mlflow_analysis.ipynb`: experiments, best runs, registered models.
- `06_streaming_demo.ipynb`: chronological replay and streaming predictions.

## Notes on Anomalib

Anomalib is image-anomaly focused. This project documents it as a benchmark option but does not force image-first models like PatchCore onto raw tabular AML transactions. The main emphasis remains custom tabular and graph implementations.
