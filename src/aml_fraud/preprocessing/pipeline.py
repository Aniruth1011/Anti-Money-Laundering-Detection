from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aml_fraud.preprocessing.schema import AMLColumns

 
@dataclass
class AMLPreprocessor:
    columns: AMLColumns = AMLColumns()

    def load(self, transactions_path: str | Path, accounts_path: str | Path | None = None):
        transactions = pd.read_csv(transactions_path)
        accounts = pd.read_csv(accounts_path) if accounts_path else None
        return self.transform(transactions, accounts)

    def transform(self, transactions: pd.DataFrame, accounts: pd.DataFrame | None = None):
        df = transactions.copy()
        df = self._normalize_transaction_columns(df)
        df = self._add_time_features(df)
        if accounts is not None and not accounts.empty:
            df = self._merge_accounts(df, accounts)
        return df.sort_values(self.columns.timestamp).reset_index(drop=True)

    def _normalize_transaction_columns(self, df: pd.DataFrame):
        aliases = {
            "tran_timestamp": self.columns.timestamp,
            "time": self.columns.timestamp,
            "orig_acct": self.columns.sender,
            "sender": self.columns.sender,
            "nameOrig": self.columns.sender,
            "dest_acct": self.columns.receiver,
            "bene_acct": self.columns.receiver,
            "receiver": self.columns.receiver,
            "nameDest": self.columns.receiver,
            "tx_amount": self.columns.amount,
            "base_amt": self.columns.amount,
            "type": self.columns.transaction_type,
            "tx_type": self.columns.transaction_type,
            "fraud": self.columns.label,
            "isSAR": self.columns.label,
        }
        df = df.rename(columns={key: value for key, value in aliases.items() if key in df.columns})
        df[self.columns.timestamp] = pd.to_datetime(df[self.columns.timestamp], errors="coerce")
        df[self.columns.amount] = pd.to_numeric(df[self.columns.amount], errors="coerce").fillna(0.0)
        if self.columns.label not in df.columns:
            df[self.columns.label] = 0
        return df

    def _add_time_features(self, df: pd.DataFrame):
        ts = df[self.columns.timestamp]
        df["hour"] = ts.dt.hour.fillna(0).astype(int)
        df["weekday"] = ts.dt.weekday.fillna(0).astype(int)
        df["day"] = ts.dt.day.fillna(1).astype(int)
        return df

    def _merge_accounts(self, df: pd.DataFrame, accounts: pd.DataFrame):
        accounts = accounts.copy().rename(
            columns={
                "acct_id": self.columns.account_id,
                "type": self.columns.account_type,
                "initial_deposit": self.columns.initial_balance,
                "prior_sar_count": self.columns.account_is_sar,
            }
        )
        keep = [
            self.columns.account_id,
            self.columns.account_type,
            self.columns.country,
            self.columns.initial_balance,
            self.columns.account_is_sar,
            "account_in_alert",
            "bank_id",
            "tx_behavior_id",
        ]
        available = [col for col in keep if col in accounts.columns]
        accounts = accounts[available].drop_duplicates(self.columns.account_id)
        accounts[self.columns.account_id] = accounts[self.columns.account_id].astype(str)
        df[self.columns.sender] = df[self.columns.sender].astype(str)
        df[self.columns.receiver] = df[self.columns.receiver].astype(str)
        sender_meta = accounts.add_prefix("sender_")
        receiver_meta = accounts.add_prefix("receiver_")
        df = df.merge(sender_meta, left_on=self.columns.sender, right_on=f"sender_{self.columns.account_id}", how="left")
        df = df.merge(receiver_meta, left_on=self.columns.receiver, right_on=f"receiver_{self.columns.account_id}", how="left")
        return df.drop(columns=[f"sender_{self.columns.account_id}", f"receiver_{self.columns.account_id}"], errors="ignore")
