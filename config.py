from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
LOG_DIR = ROOT / "results" / "logs"
MODEL_DIR = ROOT / "models"

OBSERVATION_INTERVAL = 1  # one row per observation; wall-clock interval is undocumented
RANDOM_SEEDS = [42, 43, 44, 45, 46]
PREDICTION_WEIGHT_CANDIDATES = [0.25, 0.50, 0.75]
STREAM_COUNTS = [10, 25, 50, 100]
TRAFFIC_CONDITIONS = {"LOW": 0.45, "MEDIUM": 0.80, "HIGH": 1.10}
SPIKE_MULTIPLIER = 1.8
BANDWIDTH_CAPACITY = 100.0  # normalized simulation capacity units, not a measured Mbps value
ALPHA, BETA, GAMMA = 0.60, 0.20, 0.20
QOS_LATENCY_TARGET_MS = 100.0
BASE_LATENCY_MS = 10.0
# Finite queue and bounded queue-wait model for one-observation time steps.
# The data do not document wall-clock sample duration, so queue delay is a
# normalized simulation estimate, capped instead of extrapolated without bound.
MAX_BACKLOG_CAPACITY_MULTIPLE = 2.0
MIN_QUEUE_ALLOCATION_FRACTION = 0.01
QUEUE_DELAY_SCALE_MS = 50.0
MAX_QUEUE_DELAY_MS = 500.0
PROPOSED_MIN_EQUAL_SHARE_FRACTION = 0.10
FAIRNESS_WEIGHT_MIN = 0.80
FAIRNESS_WEIGHT_MAX = 1.25
MODEL_PARAMETERS = {"n_estimators": 150, "random_state": 42, "n_jobs": -1}
