import re
import pandas as pd


def _find_column(columns, candidates):
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def packet_rows_to_demand(frame, source_name, window_seconds=1):
    """Convert packet records into per-window bit rates; infer fields conservatively."""
    time_col = _find_column(frame.columns, ("time", "timestamp", "datetime", "date"))
    byte_col = _find_column(frame.columns, ("length", "bytes", "packetlength", "totallength", "size"))
    if time_col is None or byte_col is None:
        raise ValueError(
            f"Cannot identify timestamp and packet-byte columns in {source_name}. "
            f"Found columns: {list(frame.columns)}"
        )
    user_col = _find_column(frame.columns, ("user", "userid", "session", "sessionid", "device", "deviceid"))
    app_col = _find_column(frame.columns, ("application", "service", "category", "app"))
    data = frame.copy()
    data["_time"] = pd.to_datetime(data[time_col], errors="coerce", utc=True)
    data["_bytes"] = pd.to_numeric(data[byte_col], errors="coerce")
    data = data.dropna(subset=["_time", "_bytes"])
    data = data[data["_bytes"] >= 0].drop_duplicates().sort_values("_time")
    if data.empty:
        raise ValueError(f"No valid timestamp/traffic records remained in {source_name}.")
    data["user_id"] = data[user_col].astype(str) if user_col else source_name
    data["application"] = data[app_col].astype(str) if app_col else "unknown"
    data["timestamp"] = data["_time"].dt.floor(f"{window_seconds}s")
    demand = (data.groupby(["user_id", "application", "timestamp"], as_index=False)["_bytes"].sum())
    demand["demand_bps"] = demand["_bytes"] * 8 / window_seconds
    return demand[["timestamp", "user_id", "application", "demand_bps"]]


def bitrate_rows_to_demand(frame, stream_id, observation_interval=1):
    """Keep an existing DL/UL bitrate series in source units and row order."""
    required = {"DL_bitrate", "UL_bitrate"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected columns {sorted(required)} for bitrate traces; found {list(frame.columns)}")
    data = frame[["DL_bitrate", "UL_bitrate"]].copy()
    data["DL_bitrate"] = pd.to_numeric(data["DL_bitrate"], errors="coerce")
    data["UL_bitrate"] = pd.to_numeric(data["UL_bitrate"], errors="coerce")
    # Keep paired observations only; preserve their original row positions as time_step.
    data["time_step"] = range(len(data))
    data = data.dropna(subset=["DL_bitrate", "UL_bitrate"])
    data = data[(data.DL_bitrate >= 0) & (data.UL_bitrate >= 0)].copy()
    data["stream_id"] = stream_id
    data["demand"] = data["DL_bitrate"]
    data["observation_interval"] = observation_interval
    if data.empty:
        raise ValueError(f"No valid paired DL/UL bitrate rows remained in {stream_id}.")
    return data[["stream_id", "time_step", "DL_bitrate", "UL_bitrate", "demand", "observation_interval"]]


def rows_to_demand(frame, source_name, observation_interval=1):
    """Select the time-series schema when present, otherwise use packet rows."""
    if {"DL_bitrate", "UL_bitrate"}.issubset(frame.columns):
        return bitrate_rows_to_demand(frame, source_name, observation_interval)
    packets = packet_rows_to_demand(frame, source_name, observation_interval)
    packets = packets.sort_values(["user_id", "application", "timestamp"]).copy()
    packets["stream_id"] = packets["user_id"].astype(str) + ":" + packets["application"].astype(str)
    packets["time_step"] = packets.groupby("stream_id").cumcount()
    packets["DL_bitrate"] = packets["demand_bps"]
    packets["UL_bitrate"] = 0.0
    packets["demand"] = packets["DL_bitrate"]
    packets["observation_interval"] = observation_interval
    return packets[["stream_id", "time_step", "DL_bitrate", "UL_bitrate", "demand", "observation_interval"]]
