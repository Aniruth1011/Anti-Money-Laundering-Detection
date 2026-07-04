from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AMLColumns:
    timestamp: str = "timestamp"
    sender: str = "sender_account"
    receiver: str = "receiver_account"
    amount: str = "amount"
    transaction_type: str = "transaction_type"
    label: str = "is_fraud"
    account_id: str = "account_id"
    account_type: str = "account_type"
    country: str = "country"
    initial_balance: str = "initial_balance"
    account_is_sar: str = "account_is_sar"
