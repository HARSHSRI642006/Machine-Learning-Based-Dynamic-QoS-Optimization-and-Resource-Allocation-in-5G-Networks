from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import (TABLE_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR, STREAM_COUNTS,
                     RANDOM_SEEDS, TRAFFIC_CONDITIONS, SPIKE_MULTIPLIER,
                     BANDWIDTH_CAPACITY, PREDICTION_WEIGHT_CANDIDATES)
from .simulator import run_method
from .predictor import regenerate_prediction_figures

METRICS = ["throughput", "latency", "packet_loss", "jitter", "qos_satisfaction", "jain_fairness", "resource_utilization"]
BASELINES = ["Equal Share", "Round Robin", "Proportional Fair"]
PREDICTION_ONLY = "Prediction Only (lambda=1.00)"
CURRENT_ONLY = "Current Demand Only (lambda=0.00)"


def _blend_method_name(weight):
    return f"Blended Demand (lambda={float(weight):.2f})"


def _all_methods():
    return BASELINES + [_blend_method_name(value) for value in PREDICTION_WEIGHT_CANDIDATES] + [PREDICTION_ONLY, CURRENT_ONLY]


def _profiles(predictions, users, condition, seed):
    rng = np.random.default_rng(seed)
    data = predictions[predictions.model == "RandomForestRegressor"]
    streams = {name: group.sort_values("time_step") for name, group in data.groupby("stream_id")}
    stream_names = sorted(streams)
    if not stream_names: raise ValueError("No held-out stream predictions available for simulation.")
    # Each row below is always sourced from one stream's own held-out sequence.
    # Relative observation indexes define a controlled concurrency scenario;
    # rows from distinct streams are never concatenated into a pooled sequence.
    length = min(len(streams[name]) for name in stream_names)
    if length < 3: raise ValueError("Too few held-out observations in the available streams.")
    real_count = min(users, len(stream_names))
    selected = sorted(rng.choice(stream_names, size=real_count, replace=False).tolist())
    demand_profiles, predicted_profiles, manifest = [], [], []
    for stream_id in selected:
        frame = streams[stream_id]
        demand_profiles.append(frame.current_demand.to_numpy(dtype=float)[:length].copy())
        predicted_profiles.append(frame.predicted_demand.to_numpy(dtype=float)[:length].copy())
        manifest.append({"stream_id": str(stream_id), "source_stream_id": str(stream_id), "synthetic": False, "circular_offset": 0})
    for synthetic_index in range(users - real_count):
        source_id = str(rng.choice(stream_names))
        frame = streams[source_id]
        offset = int(rng.integers(0, length))
        # Rotate within this source stream only; actual and forecast use the
        # same offset so each synthetic stream retains its paired demand.
        demand_profiles.append(np.roll(frame.current_demand.to_numpy(dtype=float)[:length], offset))
        predicted_profiles.append(np.roll(frame.predicted_demand.to_numpy(dtype=float)[:length], offset))
        manifest.append({"stream_id": f"synthetic_{synthetic_index + 1:03d}",
                         "source_stream_id": source_id, "synthetic": True, "circular_offset": offset})
    demand = np.vstack(demand_profiles)
    predicted = np.vstack(predicted_profiles)
    # Normalize scenario load to the configured capacity; source bitrate units remain unknown.
    target_load = TRAFFIC_CONDITIONS.get(condition, 0.80)
    mean_total = max(float(demand.sum(axis=0).mean()), 1e-9)
    scale = target_load * BANDWIDTH_CAPACITY / mean_total
    demand *= scale; predicted *= scale
    if condition == "SPIKE":
        split = demand.shape[1] // 2
        demand[:, split:] *= SPIKE_MULTIPLIER
        predicted[:, split:] *= SPIKE_MULTIPLIER
    return demand, predicted, manifest


def _validation_lambda_scores(validation_predictions):
    required = {"actual_demand", "current_demand", "predicted_demand"}
    if not required.issubset(validation_predictions.columns):
        raise ValueError(f"Validation predictions missing required columns: {sorted(required - set(validation_predictions.columns))}")
    actual = validation_predictions.actual_demand.to_numpy(dtype=float)
    current = validation_predictions.current_demand.to_numpy(dtype=float)
    predicted = validation_predictions.predicted_demand.to_numpy(dtype=float)
    return {float(weight): float(np.mean(np.abs(actual - (weight * predicted + (1 - weight) * current))))
            for weight in [0.0, *PREDICTION_WEIGHT_CANDIDATES, 1.0]}


def run_experiments(prediction_results, validation_predictions, selected_lambda, validation_scores):
    TABLE_DIR.mkdir(parents=True, exist_ok=True); FIGURE_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Verify the pre-test choice using validation rows only; never reselect
    # based on any test-scenario outcomes.
    validation_check = _validation_lambda_scores(validation_predictions)
    if float(selected_lambda) not in PREDICTION_WEIGHT_CANDIDATES:
        raise ValueError("Selected lambda is outside the predeclared validation candidate set.")
    if any(not np.isclose(validation_check[float(k)], validation_scores[float(k)]) for k in validation_scores):
        raise ValueError("Validation lambda scores changed between selection and experiment evaluation.")
    if min(PREDICTION_WEIGHT_CANDIDATES, key=lambda weight: (validation_check[float(weight)], weight)) != float(selected_lambda):
        raise ValueError("Selected lambda is not the minimum validation-MAE candidate.")
    runs, ablations, manifests = [], [], []
    real_stream_count = prediction_results.stream_id.nunique()
    for count in STREAM_COUNTS:
        for condition in [*TRAFFIC_CONDITIONS, "SPIKE"]:
            for seed in RANDOM_SEEDS:
                demand, forecasts, stream_manifest = _profiles(prediction_results, count, condition, seed)
                scenario_type = "real_application_traces" if count <= real_stream_count else "synthetic_scalability_scenario_derived_from_real_traffic"
                real_used = sum(not item["synthetic"] for item in stream_manifest)
                synthetic_used = count - real_used
                source_stream_ids = ";".join(item["source_stream_id"] for item in stream_manifest)
                manifests.extend({"streams": count, "stream_scenario": scenario_type,
                                  "traffic_condition": condition, "seed": seed, **item}
                                 for item in stream_manifest)
                priorities = np.ones(count); priorities[::3] = 1.5
                method_weights = {"Equal Share": None, "Round Robin": None, "Proportional Fair": None,
                                  **{_blend_method_name(value): value for value in PREDICTION_WEIGHT_CANDIDATES},
                                  PREDICTION_ONLY: 1.0, CURRENT_ONLY: 0.0}
                for method, weight in method_weights.items():
                    result = run_method(method, demand, BANDWIDTH_CAPACITY, priorities, seed,
                                        predictions=forecasts if weight is not None else None,
                                        prediction_weight=weight if weight is not None else 0.0)
                    runs.append({"streams": count, "stream_scenario": scenario_type, "real_streams": real_used,
                                 "synthetic_streams": synthetic_used, "source_stream_ids": source_stream_ids,
                                 "traffic_condition": condition, "method": method,
                                 "prediction_weight": weight, "selected_lambda": selected_lambda, **result})
                ablation_specs = [("FULL BLENDED MODEL", selected_lambda, None),
                                  ("PREDICTION ONLY", 1.0, None), ("NO PREDICTION", 0.0, None),
                                  ("NO FAIRNESS", selected_lambda, "NO FAIRNESS"),
                                  ("NO PRIORITY", selected_lambda, "NO PRIORITY"),
                                  ("SINGLE OBJECTIVE", selected_lambda, "SINGLE OBJECTIVE")]
                for label, weight, ablation_flag in ablation_specs:
                    result = run_method(_blend_method_name(weight), demand, BANDWIDTH_CAPACITY,
                                        priorities, seed, predictions=forecasts,
                                        prediction_weight=weight, ablation=ablation_flag)
                    ablations.append({"streams": count, "stream_scenario": scenario_type, "real_streams": real_used,
                                      "synthetic_streams": synthetic_used, "source_stream_ids": source_stream_ids,
                                      "traffic_condition": condition, "seed": seed, "method": label,
                                      "prediction_weight": weight, "selected_lambda": selected_lambda, **result})
    all_results = pd.DataFrame(runs); ablation_results = pd.DataFrame(ablations)
    all_results.to_csv(TABLE_DIR / "all_results.csv", index=False)
    ablation_results.to_csv(TABLE_DIR / "ablation_results.csv", index=False)
    pd.DataFrame(manifests).to_csv(TABLE_DIR / "scenario_manifest.csv", index=False)
    summary = all_results.groupby(["streams", "stream_scenario", "traffic_condition", "method"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{stat}_{metric}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    # Keep requested column convention mean_metric / std_metric.
    summary = summary.rename(columns={f"mean_{m}": f"mean_{m}" for m in METRICS})
    summary.to_csv(TABLE_DIR / "summary_results.csv", index=False)
    lambda_rows = []
    for weight in [0.0, *PREDICTION_WEIGHT_CANDIDATES, 1.0]:
        method = CURRENT_ONLY if weight == 0 else PREDICTION_ONLY if weight == 1 else _blend_method_name(weight)
        test_means = all_results[all_results.method == method][METRICS].mean()
        lambda_rows.append({"lambda": weight, "validation_metric": validation_scores[float(weight)],
                            "validation_metric_name": "MAE (source bitrate units)",
                            "selection_candidate": weight in PREDICTION_WEIGHT_CANDIDATES,
                            "selected": bool(weight == selected_lambda),
                            **{f"test_{metric}": float(test_means[metric]) for metric in METRICS}})
    lambda_comparison = pd.DataFrame(lambda_rows)
    lambda_comparison.to_csv(TABLE_DIR / "lambda_comparison.csv", index=False)
    _statistical_tests(all_results, selected_lambda)
    _plot_experiments(all_results, ablation_results, selected_lambda)
    _plot_blending_comparisons(all_results, selected_lambda)
    regenerate_prediction_figures(prediction_results)
    config_path = LOG_DIR / "experiment_config.txt"
    try:
        import json
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        config = {}
    config["lambda_selection"] = {"candidate_values": PREDICTION_WEIGHT_CANDIDATES,
                                  "criterion": "validation MAE of lambda*RF_next + (1-lambda)*current_demand",
                                  "validation_mae_by_lambda": {str(key): value for key, value in validation_scores.items()},
                                  "selected_lambda": float(selected_lambda),
                                  "test_data_used_for_selection": False,
                                  "test_evaluation_started_after_lambda_freeze": True}
    config["selected_lambda"] = float(selected_lambda)
    config["simulation_input"] = "current observed demand is the simulated offered load; next-step actual labels are excluded from allocator inputs"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return all_results, summary, ablation_results, lambda_comparison, selected_lambda


def _statistical_tests(frame, selected_lambda):
    rows = []
    proposed = _blend_method_name(selected_lambda)
    comparators = BASELINES + [PREDICTION_ONLY, CURRENT_ONLY]
    for (streams, condition, metric), group in frame.melt(id_vars=["streams", "traffic_condition", "seed", "method"], value_vars=METRICS, var_name="metric", value_name="value").groupby(["streams", "traffic_condition", "metric"]):
        pivot = group.pivot(index="seed", columns="method", values="value")
        if proposed not in pivot: continue
        for baseline in comparators:
            a, b = pivot[proposed].dropna(), pivot[baseline].dropna()
            common = a.index.intersection(b.index); a, b = a.loc[common], b.loc[common]
            if len(a) < 2: continue
            differences = a.to_numpy() - b.to_numpy()
            if np.std(differences, ddof=1) == 0:
                ttest_p = np.nan
                ttest_status = "undefined_constant_paired_difference"
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    test = stats.ttest_rel(a, b)
                ttest_p = float(test.pvalue)
                ttest_status = "ok" if np.isfinite(ttest_p) else "undefined_numeric_precision"
            if np.allclose(differences, 0):
                wilcoxon_p = np.nan
                wilcoxon_status = "undefined_no_nonzero_paired_differences"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        wilcoxon = stats.wilcoxon(a, b)
                    wilcoxon_p = float(wilcoxon.pvalue)
                    wilcoxon_status = "ok" if np.isfinite(wilcoxon_p) else "undefined_numeric_precision"
                except ValueError:
                    wilcoxon_p = np.nan
                    wilcoxon_status = "not_applicable"
            diff = a - b; sem = stats.sem(diff); ci = stats.t.interval(.95, len(diff)-1, loc=diff.mean(), scale=sem) if sem else (diff.mean(), diff.mean())
            rows.append({"streams": streams, "traffic_condition": condition, "metric": metric, "proposed_vs": baseline,
                         "n_pairs": len(a), "mean_difference": float(diff.mean()), "ci95_low": float(ci[0]),
                         "ci95_high": float(ci[1]), "ttest_p": ttest_p, "ttest_status": ttest_status,
                         "wilcoxon_p": wilcoxon_p, "wilcoxon_status": wilcoxon_status,
                         "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else np.nan,
                         "alpha": .05})
    output = pd.DataFrame(rows)
    # Holm family-wise correction across the full set of reported comparisons,
    # calculated separately for paired t-tests and Wilcoxon tests.
    for column in ["ttest_p", "wilcoxon_p"]:
        adjusted = np.full(len(output), np.nan)
        p = output[column].to_numpy(dtype=float)
        finite = np.isfinite(p)
        if finite.any():
            indexes = np.flatnonzero(finite)
            order = np.argsort(p[finite])
            sorted_p = p[finite][order]
            m = len(sorted_p)
            sorted_adjusted = np.minimum(1.0, np.maximum.accumulate(sorted_p * (m - np.arange(m))))
            adjusted[indexes[order]] = sorted_adjusted
        output[column.replace("_p", "_holm_p")] = adjusted
    output.to_csv(TABLE_DIR / "statistical_tests.csv", index=False)


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURE_DIR / f"{name}.png", dpi=200); fig.savefig(FIGURE_DIR / f"{name}.pdf"); plt.close(fig)


def _plot_experiments(results, ablation, selected_lambda):
    for metric, label in zip(METRICS, ["Throughput (normalized units)", "Latency (ms)", "Packet loss (%)", "Jitter (ms)", "QoS satisfaction (%)", "Jain fairness", "Resource utilization (%)"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, part in results[results.traffic_condition == "HIGH"].groupby("method"):
            stat = part.groupby("streams")[metric].mean()
            ax.plot(stat.index, stat.values, marker="o", label=method)
        ax.set(title=f"{label} vs traffic stream count (high load)", xlabel="Traffic streams (real and synthetic)", ylabel=label); ax.legend(fontsize=8)
        _save(fig, metric + "_vs_streams")
    spike = results[results.traffic_condition == "SPIKE"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, part in spike.groupby("method"):
        ax.plot(part.groupby("streams").throughput.mean(), marker="o", label=method)
    ax.set(title="Throughput under traffic spike", xlabel="Traffic streams", ylabel="Throughput (normalized simulation units)"); ax.legend(fontsize=8); _save(fig, "traffic_spike")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, part in ablation.groupby("method"):
        ax.plot(part.groupby("streams").throughput.mean(), marker="o", label=method)
    ax.set(title="Ablation comparison", xlabel="Traffic streams", ylabel="Throughput (normalized simulation units)"); ax.legend(fontsize=8); _save(fig, "ablation_comparison")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, part in results[results.traffic_condition == "HIGH"].groupby("method"):
        ax.plot(part.groupby("streams").qos_satisfaction.mean(), marker="o", label=method)
    ax.set(title=f"Blended (lambda={selected_lambda:.2f}) and baseline QoS comparison", xlabel="Traffic streams", ylabel="QoS satisfaction (%)"); ax.legend(fontsize=8); _save(fig, "proposed_vs_baselines")


def _plot_blending_comparisons(results, selected_lambda):
    candidate_methods = [_blend_method_name(value) for value in PREDICTION_WEIGHT_CANDIDATES]
    candidate = results[results.method.isin(candidate_methods)]
    metrics = ["throughput", "latency", "packet_loss", "jitter", "qos_satisfaction", "jain_fairness"]
    labels = ["Throughput (normalized units)", "Latency (ms)", "Packet loss (%)", "Jitter (ms)", "QoS satisfaction (%)", "Jain fairness"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    means = candidate.groupby("method")[metrics].mean()
    for ax, metric, label in zip(axes.flat, metrics, labels):
        vals = [means.loc[method, metric] for method in candidate_methods]
        colors = ["tab:orange" if value == selected_lambda else "tab:blue" for value in PREDICTION_WEIGHT_CANDIDATES]
        ax.bar(PREDICTION_WEIGHT_CANDIDATES, vals, color=colors, width=.12)
        ax.set(title=label, xlabel="Prediction weight (lambda)")
        ax.set_xticks(PREDICTION_WEIGHT_CANDIDATES)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle(f"Validation candidates on test scenarios; selected lambda={selected_lambda:.2f}")
    _save(fig, "blending_lambda_comparison")

    compare_methods = [PREDICTION_ONLY, _blend_method_name(selected_lambda), CURRENT_ONLY]
    compare_labels = ["Prediction only (lambda=1)", f"Blended (lambda={selected_lambda:.2f})", "Current demand only (lambda=0)"]
    subset = results[results.method.isin(compare_methods)]
    means = subset.groupby("method")[metrics].mean()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        ax.bar(compare_labels, [means.loc[method, metric] for method in compare_methods],
               color=["tab:blue", "tab:orange", "tab:green"])
        ax.set(title=label)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Prediction only, validation-selected blend, and current demand only")
    _save(fig, "prediction_only_vs_blended_vs_current")
