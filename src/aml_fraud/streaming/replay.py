from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class ReplayEvent:
    transaction_id: int
    payload: dict
    event_time: pd.Timestamp


class ReplayEngine:
    def __init__(self, df: pd.DataFrame, timestamp_column: str = "timestamp", speed: float = 1.0):
        self.df = df.sort_values(timestamp_column).reset_index(drop=True)
        self.timestamp_column = timestamp_column
        self.speed = max(speed, 0.0)

    def events(self):
        previous_time = None
        for idx, row in self.df.iterrows():
            event_time = pd.to_datetime(row[self.timestamp_column])
            if previous_time is not None and self.speed > 0:
                delay = max((event_time - previous_time).total_seconds() / self.speed, 0)
                time.sleep(min(delay, 1.0))
            previous_time = event_time
            yield ReplayEvent(idx, row.to_dict(), event_time)
