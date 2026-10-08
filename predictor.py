import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import MODEL_PARAMETERS, MODEL_DIR, TABLE_DIR, FIGURE_DIR, PREDICTION_WEIGHT_CANDIDATES
from .features import FEATURE_COLUMNS, make_features


def train_and_evaluate(demand):
    warnings.filterwarnings(
        "ignore",
        message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
        category=UserWarning,
    )
    features = make_features(demand)
    if len(features) < 10:
        raise ValueError("Not enough sequential observations after feature construction (need at least 10).")
    train_parts, validation_parts, test_parts = [], [], []
    for _, stream in features.groupby("stream_id", sort=False):
        stream = stream.sort_values("time_step")
        n = len(stream); train_end = int(n * .70); validation_end = int(n * .85)
        if train_end < 1 or validation_end <= train_end or validation_end >= n:
            continue
        train_parts.append(stream.iloc[:train_end])
        validation_parts.append(stream.iloc[train_end:validation_end])
        test_parts.append(stream.iloc[validation_end:])
    if not train_parts or not validation_parts or not test_parts:
        raise ValueError("Not enough sequential observations for chronological 70/15/15 splits.")
    train = pd.concat(train_parts); validation = pd.concat(validation_parts)
    max_depth = max(4, min(20, int(np.log2(len(train)))))
    model_parameters = MODEL_PARAMETERS | {"max_depth": max_depth}
    model = RandomForestRegressor(**model_parameters)
    model.fit(train[FEATURE_COLUMNS], train["next_DL_bitrate"])
    # Generate validation forecasts from the train-only model and freeze
    # lambda before materializing or predicting the test partition.
    validation_predictions = model.predict(validation[FEATURE_COLUMNS])
    validation_output = validation[["stream_id", "target_time_step", "current_DL"]].copy()
    validation_output = validation_output.rename(columns={"target_time_step": "time_step", "current_DL": "current_demand"})
    validation_output["actual_demand"] = validation["next_DL_bitrate"].to_numpy()
    validation_output["predicted_demand"] = np.maximum(validation_predictions, 0)
    validation_output["model"] = "RandomForestRegressor"
    selected_lambda, validation_scores = select_lambda_from_validation(validation_output)

    # Final test evaluation begins only after the validation-only choice above.
    test = pd.concat(test_parts)
    predictions = model.predict(test[FEATURE_COLUMNS])
    persistence = test["current_DL"].to_numpy()
    actual = test["next_DL_bitrate"].to_numpy()
    output = test[["stream_id", "target_time_step", "current_DL"]].copy()
    output = output.rename(columns={"target_time_step": "time_step", "current_DL": "current_demand"})
    output["actual_demand"] = actual
    output["predicted_demand"] = np.maximum(predictions, 0)
    output["model"] = "RandomForestRegressor"
    baseline = output.copy(); baseline["predicted_demand"] = persistence; baseline["model"] = "Persistence"
    output = pd.concat([output, baseline], ignore_index=True)
    summaries = []
    for name, pred in (("RandomForestRegressor", np.maximum(predictions, 0)), ("Persistence", persistence)):
        summaries.append({"model": name, "MAE": mean_absolute_error(actual, pred),
                          "RMSE": np.sqrt(mean_squared_error(actual, pred)), "R2": r2_score(actual, pred)})
    MODEL_DIR.mkdir(parents=True, exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, MODEL_DIR / "random_forest.joblib")
    (MODEL_DIR / "model_parameters.txt").write_text(str(model_parameters), encoding="utf-8")
    output.to_csv(TABLE_DIR / "prediction_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(TABLE_DIR / "prediction_summary.csv", index=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    forest = output[output.model == "RandomForestRegressor"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    selected_stream = forest.stream_id.iloc[0]
    sample = forest[forest.stream_id == selected_stream].sort_values("time_step").head(500)
    ax.plot(sample.time_step, sample.actual_demand, label="Actual")
    ax.plot(sample.time_step, sample.predicted_demand, label="Random Forest")
    ax.set(title=f"Actual and predicted DL demand: {selected_stream}", xlabel="Sequential time step", ylabel="DL bitrate (source units; unit undocumented)"); ax.legend()
    _save_figure(fig, "actual_vs_predicted")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(forest.actual_demand - forest.predicted_demand, bins=40)
    ax.set(title="Prediction error distribution", xlabel="Actual - predicted (source units)", ylabel="Observations")
    _save_figure(fig, "prediction_error_distribution")
    for metric, title in (("MAE", "MAE comparison"), ("RMSE", "RMSE comparison")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(pd.DataFrame(summaries).model, pd.DataFrame(summaries)[metric])
        ax.set(title=title, xlabel="Model", ylabel=f"{metric} (source bitrate units)"); ax.tick_params(axis="x", rotation=15)
        _save_figure(fig, metric.lower() + "_comparison")
    return model, output, pd.DataFrame(summaries), validation_output, selected_lambda, validation_scores


def select_lambda_from_validation(validation_predictions):
    required = {"actual_demand", "current_demand", "predicted_demand"}
    if not required.issubset(validation_predictions.columns):
        raise ValueError(f"Validation predictions missing required columns: {sorted(required - set(validation_predictions.columns))}")
    actual = validation_predictions.actual_demand.to_numpy(dtype=float)
    current = validation_predictions.current_demand.to_numpy(dtype=float)
    predicted = validation_predictions.predicted_demand.to_numpy(dtype=float)
    scores = {float(weight): float(np.mean(np.abs(actual - (weight * predicted + (1 - weight) * current))))
              for weight in [0.0, *PREDICTION_WEIGHT_CANDIDATES, 1.0]}
    selected = min(PREDICTION_WEIGHT_CANDIDATES, key=lambda weight: (scores[float(weight)], weight))
    return float(selected), scores


def _save_figure(fig, name):
    fig.tight_layout(); fig.savefig(FIGURE_DIR / f"{name}.png", dpi=200); fig.savefig(FIGURE_DIR / f"{name}.pdf"); plt.close(fig)


def regenerate_prediction_figures(output):
    """Refresh prediction plots from the retained measured held-out predictions."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    forest = output[output.model == "RandomForestRegressor"]
    if forest.empty:
        raise ValueError("No Random Forest held-out rows available for prediction figures.")
    selected_stream = sorted(forest.stream_id.unique())[0]
    sample = forest[forest.stream_id == selected_stream].sort_values("time_step").head(500)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sample.time_step, sample.actual_demand, label="Actual")
    ax.plot(sample.time_step, sample.predicted_demand, label="Random Forest")
    ax.set(title=f"Actual and predicted DL demand: {selected_stream}", xlabel="Sequential time step", ylabel="DL bitrate (source units; unit undocumented)"); ax.legend()
    _save_figure(fig, "actual_vs_predicted")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(forest.actual_demand - forest.predicted_demand, bins=40)
    ax.set(title="Prediction error distribution", xlabel="Actual - predicted (source units)", ylabel="Observations")
    _save_figure(fig, "prediction_error_distribution")
    summary = output.groupby("model").apply(lambda g: pd.Series({
        "MAE": mean_absolute_error(g.actual_demand, g.predicted_demand),
        "RMSE": np.sqrt(mean_squared_error(g.actual_demand, g.predicted_demand))
    }), include_groups=False).reset_index()
    for metric, title in (("MAE", "MAE comparison"), ("RMSE", "RMSE comparison")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(summary.model, summary[metric])
        ax.set(title=title, xlabel="Model", ylabel=f"{metric} (source bitrate units)"); ax.tick_params(axis="x", rotation=15)
        _save_figure(fig, metric.lower() + "_comparison")
