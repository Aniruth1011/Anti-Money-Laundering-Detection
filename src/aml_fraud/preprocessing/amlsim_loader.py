from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TRUTHY = {"true", "1", "yes"}

@dataclass
class AMLSimDataset:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    alert_accounts: pd.DataFrame
    alert_transactions: pd.DataFrame
    sar_accounts: pd.DataFrame
    tx_log: pd.DataFrame


def load_amlsim_output(source_dir: str | Path):
    source = Path(source_dir)
    return AMLSimDataset(
        _read_required(source / "accounts.csv"),
        _read_required(source / "transactions.csv"),
        _read_required(source / "alert_accounts.csv"),
        _read_required(source / "alert_transactions.csv"),
        _read_required(source / "sar_accounts.csv"),
        _read_required(source / "tx_log.csv"),
    )

def build_training_tables(source_dir: str | Path):
    dataset = load_amlsim_output(source_dir)
    accounts = normalize_accounts(dataset.accounts, dataset.alert_accounts, dataset.sar_accounts)
    transactions = normalize_transactions(dataset.transactions)
    transactions = add_tx_log_features(transactions, dataset.tx_log)
    transactions = add_alert_features(transactions, dataset.alert_transactions, dataset.alert_accounts, dataset.sar_accounts)
    return transactions, accounts


def normalize_accounts(accounts: pd.DataFrame, alert_accounts: pd.DataFrame, sar_accounts: pd.DataFrame):
    frame = accounts.rename(
        columns={
            "acct_id": "account_id",
            "type": "account_type",
            "prior_sar_count": "account_is_sar",
            "initial_deposit": "initial_balance",
        }
    ).copy()
    frame["account_id"] = frame["account_id"].astype(str)
    frame["initial_balance"] = pd.to_numeric(frame["initial_balance"], errors="coerce")
    frame["account_is_sar"] = as_bool(frame["account_is_sar"])
    frame["account_in_alert"] = frame["account_id"].isin(account_ids(alert_accounts) | account_ids(sar_accounts))
    return frame


def normalize_transactions(transactions: pd.DataFrame):
    frame = transactions.rename(
        columns={
            "tran_id": "transaction_id",
            "orig_acct": "sender_account",
            "bene_acct": "receiver_account",
            "tx_type": "transaction_type",
            "base_amt": "amount",
            "tran_timestamp": "timestamp",
        }
    ).copy()
    frame["transaction_id"] = pd.to_numeric(frame["transaction_id"], errors="coerce").astype("Int64")
    frame["sender_account"] = frame["sender_account"].astype(str)
    frame["receiver_account"] = frame["receiver_account"].astype(str)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["is_sar"] = as_bool(frame["is_sar"])
    frame["alert_id"] = pd.to_numeric(frame["alert_id"], errors="coerce").fillna(-1).astype(int)
    return frame


def add_tx_log_features(transactions: pd.DataFrame, tx_log: pd.DataFrame):
    log = tx_log.rename(
        columns={
            "step": "simulation_step",
            "oldbalanceOrig": "sender_old_balance",
            "newbalanceOrig": "sender_new_balance",
            "oldbalanceDest": "receiver_old_balance",
            "newbalanceDest": "receiver_new_balance",
            "isSAR": "log_is_sar",
            "alertID": "log_alert_id",
        }
    ).copy()
    log["transaction_id"] = pd.RangeIndex(1, len(log) + 1)
    keep = [
        "transaction_id",
        "simulation_step",
        "sender_old_balance",
        "sender_new_balance",
        "receiver_old_balance",
        "receiver_new_balance",
        "log_is_sar",
        "log_alert_id",
    ]
    return transactions.merge(log[keep], on="transaction_id", how="left")


def add_alert_features(transactions: pd.DataFrame, alert_transactions: pd.DataFrame, alert_accounts: pd.DataFrame, sar_accounts: pd.DataFrame):
    frame = transactions.copy()
    alert_tx = normalize_alert_transactions(alert_transactions)
    frame = frame.merge(alert_tx, on="transaction_id", how="left")

    alert_account_ids = account_ids(alert_accounts)
    sar_account_ids = account_ids(sar_accounts)
    frame["sender_in_alert"] = frame["sender_account"].isin(alert_account_ids)
    frame["receiver_in_alert"] = frame["receiver_account"].isin(alert_account_ids)
    frame["sender_is_sar_account"] = frame["sender_account"].isin(sar_account_ids)
    frame["receiver_is_sar_account"] = frame["receiver_account"].isin(sar_account_ids)
    frame["is_fraud"] = fraud_label(frame)
    return frame


def normalize_alert_transactions(alert_transactions: pd.DataFrame):
    frame = alert_transactions.rename(
        columns={
            "tran_id": "transaction_id",
            "is_sar": "alert_transaction_is_sar",
        }
    ).copy()
    frame["transaction_id"] = pd.to_numeric(frame["transaction_id"], errors="coerce").astype("Int64")
    frame["alert_transaction_is_sar"] = as_bool(frame["alert_transaction_is_sar"])
    keep = ["transaction_id", "alert_type", "alert_transaction_is_sar"]
    return frame[keep].drop_duplicates("transaction_id")


def fraud_label(frame: pd.DataFrame):
    labels = frame["is_sar"] | frame["alert_id"].ge(0)
    labels |= frame["alert_transaction_is_sar"].astype("boolean").fillna(False)
    labels |= frame["sender_is_sar_account"] | frame["receiver_is_sar_account"]
    return labels.astype(int)


def account_ids(frame: pd.DataFrame):
    for column in ["acct_id", "ACCOUNT_ID", "account_id"]:
        if column in frame.columns:
            return set(frame[column].astype(str))
    return set()


def as_bool(values: pd.Series):
    return values.astype(str).str.lower().isin(TRUTHY)


def _read_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
