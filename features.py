import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["current_DL", "current_UL", "lag_DL_1", "lag_DL_2", "lag_DL_3", "lag_DL_5",
                   "moving_average_DL", "moving_std_DL", "DL_growth_rate", "UL_lag_1", "UL_lag_2"]


def make_features(demand_frame):
    rows = []
    for stream_id, group in demand_frame.groupby("stream_id", sort=False):
        group = group.sort_values("time_step").copy()
        dl = group["DL_bitrate"].astype(float)
        ul = group["UL_bitrate"].astype(float)
        group["current_DL"] = dl
        group["current_UL"] = ul
        for lag in (1, 2, 3, 5):
            group[f"lag_DL_{lag}"] = dl.shift(lag)
        group["moving_average_DL"] = dl.rolling(5, min_periods=5).mean()
        group["moving_std_DL"] = dl.rolling(5, min_periods=5).std().fillna(0)
        group["DL_growth_rate"] = dl.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        group["UL_lag_1"] = ul.shift(1)
        group["UL_lag_2"] = ul.shift(2)
        group["next_DL_bitrate"] = dl.shift(-1)
        group["target_time_step"] = group["time_step"].shift(-1)
        group["stream_id"] = stream_id
        rows.append(group)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True).dropna(subset=FEATURE_COLUMNS + ["next_DL_bitrate", "target_time_step"])
    return result
