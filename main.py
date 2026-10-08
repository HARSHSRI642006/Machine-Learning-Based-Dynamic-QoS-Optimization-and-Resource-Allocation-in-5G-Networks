import json
import sys
import hashlib
from pathlib import Path

from src.config import (RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLE_DIR, LOG_DIR,
                        FIGURE_DIR, MODEL_DIR, MODEL_PARAMETERS,
                        ALPHA, BETA, GAMMA,
                        STREAM_COUNTS, TRAFFIC_CONDITIONS, RANDOM_SEEDS,
                        OBSERVATION_INTERVAL, BANDWIDTH_CAPACITY)
from src.config import PREDICTION_WEIGHT_CANDIDATES
from src.preprocessing import rows_to_demand
from src.predictor import train_and_evaluate
from src.experiments import run_experiments


def inspect_csv(path):
    import pandas as pd
    frame = pd.read_csv(path, low_memory=False)
    numeric = {}
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        numeric[column] = {
            "numeric": bool(converted.notna().sum() == frame[column].notna().sum()),
            "minimum": float(converted.min()) if converted.notna().any() else None,
            "maximum": float(converted.max()) if converted.notna().any() else None,
            "nan_count": int(frame[column].isna().sum()),
            "nonnumeric_count": int((converted.isna() & frame[column].notna()).sum()),
        }
    return frame, {"file": path.name, "shape": list(frame.shape), "columns": list(frame.columns),
                   "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
                   "first_5_rows": frame.head(5).where(pd.notna(frame.head(5)), None).to_dict(orient="records"),
                   "column_checks": numeric}


def main(stream_filter=None, row_limit=None):
    for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_path": str(RAW_DATA_DIR), "observation_interval": OBSERVATION_INTERVAL,
        "row_limit_for_smoke_run": row_limit,
        "time_index": "row order per application file; not a wall-clock timestamp",
        "bitrate_units": "undocumented; preserved without conversion",
        "features": ["current_DL", "current_UL", "lag_DL_1/2/3/5", "moving_average_DL", "moving_std_DL", "DL_growth_rate", "UL_lag_1/2"],
        "model": "RandomForestRegressor", "model_parameters": MODEL_PARAMETERS,
        "allocation_weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
        "bandwidth_capacity_normalized_simulation_units": BANDWIDTH_CAPACITY, "traffic_stream_counts": STREAM_COUNTS,
        "traffic_conditions": TRAFFIC_CONDITIONS | {"SPIKE": "step increase"}, "seeds": RANDOM_SEEDS,
        "prediction_weight_candidates": PREDICTION_WEIGHT_CANDIDATES,
        "lambda_selection": "minimum validation MAE of the blended estimate; final test split is not used for selection",
    }
    (LOG_DIR / "experiment_config.txt").write_text(json.dumps(config, indent=2), encoding="utf-8")
    files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if stream_filter:
        files = [path for path in files if stream_filter.lower() in path.stem.lower()]
    if not files:
        raise FileNotFoundError(f"No CSV files found for stream filter {stream_filter!r} under {RAW_DATA_DIR}")
    audit = []; demand_parts = []; seen_hashes = {}
    for path in files:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        frame, details = inspect_csv(path)
        details["sha256"] = digest
        if digest in seen_hashes:
            details["duplicate_of"] = seen_hashes[digest]
            audit.append(details)
            continue
        seen_hashes[digest] = path.name
        audit.append(details)
        analysis_frame = frame.head(row_limit) if row_limit else frame
        demand_parts.append(rows_to_demand(analysis_frame, path.stem.replace(" (1)", ""), OBSERVATION_INTERVAL))
        details["rows_used_for_analysis"] = len(analysis_frame)
    (LOG_DIR / "dataset_inspection.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    pandas = __import__("pandas")
    demand = pandas.concat(demand_parts, ignore_index=True)
    demand.to_csv(PROCESSED_DATA_DIR / "windowed_demand.csv", index=False)
    observed_streams = sorted(demand.stream_id.unique().tolist())
    summary = {"unique_traffic_streams": len(observed_streams), "stream_ids": observed_streams,
               "rows_after_preprocessing": len(demand), "measured_physical_users": "not identified by these files",
               "bitrate_units": "not specified in local files or matching dataset description; raw numeric values retained",
               "duplicate_files_skipped": [item["file"] for item in audit if "duplicate_of" in item]}
    (LOG_DIR / "stream_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_files": audit, "stream_summary": summary}, indent=2, default=str))
    _, prediction_results, _, validation_predictions, selected_lambda, validation_scores = train_and_evaluate(demand)
    run_experiments(prediction_results, validation_predictions, selected_lambda, validation_scores)
    print(f"Pipeline completed. Inspected {len(files)} input files; unique streams={len(observed_streams)}; outputs are under {TABLE_DIR.parent}.")


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Run the 5G traffic prediction and allocation experiment")
        parser.add_argument("--stream", help="Optional substring filter for a single CSV filename, e.g. afreeca")
        parser.add_argument("--limit-rows", type=int, help="Limit rows per stream for a smoke run")
        args = parser.parse_args()
        main(args.stream, args.limit_rows)
    except Exception as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        raise
