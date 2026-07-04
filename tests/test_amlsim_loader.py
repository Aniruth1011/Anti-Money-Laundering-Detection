from pathlib import Path

import pandas as pd

from aml_fraud.preprocessing.amlsim_loader import build_training_tables


def test_full_amlsim_output_uses_alerts_and_tx_log(tmp_path: Path):
    pd.DataFrame(
        {
            "tran_id": [1, 2],
            "orig_acct": [10, 11],
            "bene_acct": [12, 13],
            "tx_type": ["TRANSFER", "TRANSFER"],
            "base_amt": [100.0, 200.0],
            "tran_timestamp": ["2017-01-01T00:00:00Z", "2017-01-01T00:00:00Z"],
            "is_sar": [False, False],
            "alert_id": [-1, 4],
        }
    ).to_csv(tmp_path / "transactions.csv", index=False)
    pd.DataFrame(
        {
            "acct_id": [10, 11, 12, 13],
            "type": ["I", "I", "B", "B"],
            "country": ["US", "US", "US", "GB"],
            "initial_deposit": [1000, 2000, 3000, 4000],
            "prior_sar_count": [False, False, False, True],
        }
    ).to_csv(tmp_path / "accounts.csv", index=False)
    pd.DataFrame({"alert_id": [4], "alert_type": ["fan_in"], "acct_id": [11], "is_sar": [True]}).to_csv(
        tmp_path / "alert_accounts.csv", index=False
    )
    pd.DataFrame(
        {
            "alert_id": [4],
            "alert_type": ["fan_in"],
            "is_sar": [True],
            "tran_id": [2],
            "orig_acct": [11],
            "bene_acct": [13],
            "tx_type": ["TRANSFER"],
            "base_amt": [200.0],
            "tran_timestamp": ["2017-01-01T00:00:00Z"],
        }
    ).to_csv(tmp_path / "alert_transactions.csv", index=False)
    pd.DataFrame({"ALERT_ID": [4], "ACCOUNT_ID": [13], "IS_SAR": ["YES"]}).to_csv(tmp_path / "sar_accounts.csv", index=False)
    pd.DataFrame(
        {
            "step": [0, 0],
            "type": ["TRANSFER", "TRANSFER"],
            "amount": [100.0, 200.0],
            "nameOrig": [10, 11],
            "oldbalanceOrig": [1000.0, 2000.0],
            "newbalanceOrig": [900.0, 1800.0],
            "nameDest": [12, 13],
            "oldbalanceDest": [3000.0, 4000.0],
            "newbalanceDest": [3100.0, 4200.0],
            "isSAR": [0, 1],
            "alertID": [-1, 4],
        }
    ).to_csv(tmp_path / "tx_log.csv", index=False)

    transactions, accounts = build_training_tables(tmp_path)

    assert transactions["is_fraud"].tolist() == [0, 1]
    assert "sender_old_balance" in transactions.columns
    assert transactions.loc[1, "alert_type"] == "fan_in"
    assert accounts["account_in_alert"].sum() == 2
