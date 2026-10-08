# Machine Learning-Based Dynamic QoS Optimization and Resource Allocation in 5G Networks

## Objective

This project studies whether a one-step traffic forecast can help a simple bandwidth allocator serve changing user demand. It combines a Random Forest forecast, four allocation policies, queue-based performance measures, ablation runs, and paired statistical comparisons. The Python model is a resource-allocation simulation, not a physical 5G or radio propagation simulator.

## Dataset

The available files under `data/raw/` are application bitrate time-series CSVs with `DL_bitrate` and `UL_bitrate` columns. The file shapes match the application lengths described in a published analysis of the 5G traffic collection ([dataset description and lengths](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1477382/full)). The downloaded CSVs do not carry a unit label, and the published description we could verify does not state the unit. Therefore, preprocessing preserves the numeric source values without converting or labeling them as Mbps/Kbps.

The directory contains 16 CSV files. The two Naver NOW files are byte-identical, leaving 15 unique application streams and 1,182,637 rows after preprocessing. Afreeca has 72,840 rows; its observed DL range is 0–1,706,516 and UL range is 0–36,325 in the source's unspecified bitrate units. All Afreeca values are numeric and have no NaNs. Battleground has two missing UL rows; those paired observations are dropped.

Each distinct application file is treated as one traffic stream/session, not one real network user. Exact duplicate files are detected by SHA-256 and only the first copy is used. In the current folder, `navernow_dataset.csv` and `navernow_dataset (1).csv` are byte-identical, so the duplicate is excluded from analysis. No physical user count can be inferred from these files.

## Setup and run

```powershell
python -m pip install -r requirements.txt
python main.py
```

The pipeline writes a dataset audit and its configuration before modeling. To run a one-stream check, use `python main.py --stream afreeca`; the full directory run is `python main.py`.

## Structure

```text
data/raw/                 Application bitrate time-series CSVs
data/processed/           Preprocessed sequential observations
models/                   Trained Random Forest
results/tables/            Prediction, simulation, ablation, and test tables
results/figures/           PNG and PDF figures
results/logs/               Configuration and input audit
src/                       Loading, features, prediction, allocation, metrics
main.py
```

## Preprocessing and prediction

When both bitrate columns exist, the row order is preserved, `time_step` is set to the row number, and `demand` is `DL_bitrate`. `UL_bitrate` remains separate. The row interval is configurable as `OBSERVATION_INTERVAL`, but the value is a row-level setting only: the data documentation available here does not confirm that rows are one second apart. No clock timestamps are invented. Invalid/non-numeric or negative DL/UL rows are removed together to keep the feature pairs aligned; the original row number is retained, so skipped rows leave gaps in `time_step`.

Features use only current and previous values: `current_DL`, `current_UL`, DL lags 1/2/3/5, five-observation DL mean and standard deviation, DL growth rate, and UL lags 1/2. The target is the next observed `DL_bitrate`. Each stream is split chronologically into 70% training, 15% validation, and 15% testing. No scaler is used. The Random Forest is trained only on training rows; persistence predicts the next DL bitrate using the current DL bitrate.

## Allocation and metrics

The policies are Equal Share, Round Robin, throughput-history Proportional Fair, and the proposed forecast/priority/urgency allocator. Round Robin uses vectorized progressive equal filling: each active stream receives the same quantum, streams that meet demand leave the active set, and the remaining capacity is shared among the rest. The proposed allocator forms `blended_demand = lambda * predicted_next_demand + (1 - lambda) * current_demand`, then allocates against blended demand plus carried backlog. It caps grants at that demand. Its fairness adjustment changes historical-service weights by at most 25% and reserves 10% of an equal-share quantum for each active stream before distributing remaining capacity by weighted allocation. It remains subject to total-capacity and nonnegative-allocation constraints.

Lambda is selected from `{0.25, 0.50, 0.75}` by minimum MAE on the chronological validation split using `lambda * RF_next + (1 - lambda) * current_DL`. The measured validation MAEs were 136,950.55, 115,352.29, and 95,543.29, respectively, so the selected and frozen value is **0.75**. Lambda 0 and 1 are evaluated as current-demand-only and prediction-only endpoints, but are not tuning candidates. Test labels are not used for lambda selection. The prediction model and features are unchanged.

The CSV dataset directly provides only application DL/UL bitrate demand. Each test simulation row uses the observed current demand as offered load; the next-step actual label is excluded from allocator inputs. The RF forecast is paired with that observed demand for the prediction-only and blended variants. This avoids giving an allocator the future test label. The queue model generates the other metrics: unmet demand joins a finite shared buffer of two capacity-observations, and allocation serves current demand plus carried backlog. Overflow beyond the per-stream buffer share is dropped and counted as packet loss. Queue delay is `10 ms + 50 ms * min(backlog / max(allocation, allocation_floor), 10)`, capped at 500 ms; an unserved nonempty stream is explicitly assigned the cap. The allocation floor is 1% of the equal-share capacity per stream. This is a bounded simulation estimate, not measured end-to-end latency: the source does not document a wall-clock observation interval or packet size. Jitter is the change in mean modeled delay between adjacent observations. QoS satisfaction uses a 100 ms delay target and at most 1% modeled overflow. Jain fairness is calculated over mean allocated rates.

## Experimental conditions and limitations

Load conditions are fixed fractions of configured normalized simulation capacity in `src/config.py` (LOW 0.45, MEDIUM 0.80, HIGH 1.10); SPIKE multiplies the second half by 1.8 after a stable first half. There are 15 distinct application streams after duplicate removal. Each 10-stream run selects 10 distinct real `stream_id` profiles. The 25/50/100-stream scenarios include all 15 real profiles and independently derive additional synthetic streams from one source profile each, with the actual demand and its matching forecast shifted together within that source stream. These are explicitly marked `synthetic_scalability_scenario_derived_from_real_traffic`; they are not additional real users. `scenario_manifest.csv` records every scenario stream, source `stream_id`, synthetic flag, and offset. Profiles are normalized to configured capacity because source bitrate units are undocumented; simulated throughput therefore uses normalized simulation units, not Mbps. Seeds [42–46] control stream selection and synthetic derivation.

These traces were collected separately by application and do not establish simultaneous multi-user network traffic. Thus combining them into a competing-stream scenario is a controlled simulation arrangement, not proof of actual concurrent users or scheduler performance. The dataset version here does not include timestamps, actual bandwidth allocations, radio conditions, packet delivery outcome, latency, loss, or QoS classes. Capacity and latency/QoS parameters are modeling assumptions and should be checked before interpreting simulation outputs.

## Outputs

The run creates `prediction_results.csv`, `prediction_summary.csv`, `all_results.csv`, `summary_results.csv`, `ablation_results.csv`, `statistical_tests.csv`, `lambda_comparison.csv`, `scenario_manifest.csv`, `experiment_config.txt`, and `dataset_inspection.json`. It also creates `blending_lambda_comparison` and `prediction_only_vs_blended_vs_current` plots. Simulation tables contain values measured from the Python model; prediction tables use the held-out time-ordered portion. Figures are saved as PNG and PDF under `results/figures/`.

The `NO PREDICTION` ablation allocates from same-step observed demand instead of the next-step forecast. The full model still uses the Random Forest next-step forecast. The statistical output includes raw and Holm-adjusted p-values; the correction is applied separately to t-tests and Wilcoxon tests across the reported comparisons.

Statistical tests pair runs by seed and configuration, report paired t-test and exact Wilcoxon p-values, Holm-adjusted p-values across the full comparison set, 95% confidence intervals for paired differences, and paired Cohen's dz. When a test is undefined, its status is recorded rather than assigning a misleading p-value. The five-seed sample is small, and normality assumptions and scenario variability remain concerns; both raw and adjusted results should be interpreted cautiously.
