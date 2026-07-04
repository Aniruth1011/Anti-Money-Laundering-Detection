import pandas as pd

from aml_fraud.features.builder import FeatureBuilder
from aml_fraud.preprocessing.pipeline import AMLPreprocessor


def test_preprocessing_and_features_join_accounts():
    transactions = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "sender_account": ["a1", "a1"],
            "receiver_account": ["a2", "a3"],
            "amount": [10.0, 20.0],
            "transaction_type": ["ACH", "WIRE"],
            "is_fraud": [0, 1],
        }
    )
    accounts = pd.DataFrame(
        {
            "account_id": ["a1", "a2", "a3"],
            "account_type": ["I", "B", "I"],
            "country": ["US", "US", "GB"],
            "initial_balance": [100.0, 200.0, 300.0],
        }
    )
    frame = AMLPreprocessor().transform(transactions, accounts)
    features = FeatureBuilder(include_graph=True).build_frame(frame)
    assert "sender_outgoing_count" in features.columns
    assert "sender_account_type" in features.columns
    assert "pagerank" in features.columns
    assert len(features) == 2
