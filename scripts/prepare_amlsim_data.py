from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from aml_fraud.preprocessing.amlsim_loader import build_training_tables

REQUIRED_FILES = [
    "accounts.csv",
    "transactions.csv",
    "alert_accounts.csv",
    "alert_transactions.csv",
    "sar_accounts.csv",
    "tx_log.csv",
]


def prepare_amlsim(source_dir: Path, output_dir: Path):
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing AMLSim output files: {missing}")

    transactions, accounts = build_training_tables(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
    accounts.to_csv(output_dir / "accounts.csv", index=False)

    for name in REQUIRED_FILES[2:]:
        shutil.copyfile(source_dir / name, output_dir / name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/AMLSim/outputs/sample")
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    prepare_amlsim(Path(args.source), Path(args.output))

if __name__ == "__main__":
    main()
