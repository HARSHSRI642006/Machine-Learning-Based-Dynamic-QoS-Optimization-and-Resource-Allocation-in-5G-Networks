from pathlib import Path
import pandas as pd


def list_input_files(folder: Path):
    patterns = ("*.csv", "*.csv.gz", "*.parquet", "*.json", "*.jsonl")
    files = sorted({path for pattern in patterns for path in folder.rglob(pattern)})
    if not files:
        raise FileNotFoundError(
            f"No supported traffic files found in {folder}. Place the dataset under data/raw/ and rerun."
        )
    return files


def inspect_file(path: Path):
    rows = 0; duplicates = 0; missing = {}; dtypes = {}; columns = None
    for frame in iter_file(path):
        rows += len(frame); duplicates += int(frame.duplicated().sum())
        if columns is None:
            columns = list(frame.columns)
            missing = {str(c): 0 for c in columns}
        for column, count in frame.isna().sum().items(): missing[str(column)] += int(count)
        for column, dtype in frame.dtypes.items(): dtypes[str(column)] = str(dtype)
    frame_columns = columns or []
    normalized = {"".join(ch.lower() for ch in str(column) if ch.isalnum()): column for column in frame_columns}
    def find(candidates):
        return next((normalized[key] for key in candidates if key in normalized), None)
    return {
        "file": str(path), "format": path.suffix.lower(), "rows": rows,
        "columns": len(frame_columns), "column_names": frame_columns,
        "data_types": dtypes, "missing_values": missing,
        "duplicate_records_within_chunks": duplicates,
        "timestamp_field_candidate": find(("time", "timestamp", "datetime", "date")),
        "user_session_field_candidate": find(("user", "userid", "session", "sessionid", "device", "deviceid")),
        "traffic_field_candidate": find(("length", "bytes", "packetlength", "totallength", "size")),
    }


def read_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=suffix == ".jsonl")
    return pd.read_csv(path, low_memory=False)


def iter_file(path: Path, chunksize=250_000):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield pd.read_parquet(path)
    elif suffix in (".json", ".jsonl"):
        if suffix == ".jsonl":
            yield from pd.read_json(path, lines=True, chunksize=chunksize)
        else:
            yield pd.read_json(path)
    else:
        yield from pd.read_csv(path, low_memory=False, chunksize=chunksize)
