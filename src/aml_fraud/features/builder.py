from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aml_fraud.graph.features import build_graph_features
from aml_fraud.preprocessing.schema import AMLColumns


@dataclass
class FeatureBuilder:
    columns: AMLColumns = AMLColumns()
    include_graph: bool = True

    def build_frame(self, df: pd.DataFrame):
        features = df.copy()
        features = self._sender_aggregates(features)
        features = self._receiver_aggregates(features)
        if self.include_graph:
            graph_features = build_graph_features(features, self.columns)
            features = features.merge(graph_features, left_on=self.columns.sender, right_index=True, how="left")
            features = features.merge(graph_features.add_prefix("receiver_"), left_on=self.columns.receiver, right_index=True, how="left")
        return features

    def split_xy(self, frame: pd.DataFrame):
        y = frame[self.columns.label].astype(int).to_numpy()
        drop = {
            self.columns.label,
            self.columns.timestamp,
            self.columns.sender,
            self.columns.receiver,
            "transaction_id",
            "alert_id",
            "log_alert_id",
            "alert_type",
            "is_sar",
            "alert_transaction_is_sar",
        }
        x = frame.drop(columns=[col for col in drop if col in frame.columns])
        return x, y

    def make_transformer(self, x: pd.DataFrame):
        categorical = x.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric = [col for col in x.columns if col not in categorical]
        return ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
            ]
        )

    def _sender_aggregates(self, df: pd.DataFrame):
        grouped = df.groupby(self.columns.sender)
        agg = grouped.agg(
            sender_outgoing_count=(self.columns.amount, "size"),
            sender_avg_outgoing_amount=(self.columns.amount, "mean"),
            sender_total_outgoing_amount=(self.columns.amount, "sum"),
            sender_unique_receivers=(self.columns.receiver, "nunique"),
        )
        return df.merge(agg, left_on=self.columns.sender, right_index=True, how="left")

    def _receiver_aggregates(self, df: pd.DataFrame):
        grouped = df.groupby(self.columns.receiver)
        agg = grouped.agg(
            receiver_incoming_count=(self.columns.amount, "size"),
            receiver_avg_incoming_amount=(self.columns.amount, "mean"),
            receiver_total_incoming_amount=(self.columns.amount, "sum"),
            receiver_unique_senders=(self.columns.sender, "nunique"),
        )
        return df.merge(agg, left_on=self.columns.receiver, right_index=True, how="left")
