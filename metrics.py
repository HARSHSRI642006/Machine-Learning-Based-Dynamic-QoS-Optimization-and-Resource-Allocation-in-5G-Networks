import numpy as np
from .config import (BASE_LATENCY_MS, MAX_BACKLOG_CAPACITY_MULTIPLE,
                     MIN_QUEUE_ALLOCATION_FRACTION, QUEUE_DELAY_SCALE_MS,
                     MAX_QUEUE_DELAY_MS, QOS_LATENCY_TARGET_MS)


def jain_fairness(values):
    values = np.maximum(np.asarray(values, dtype=float), 0)
    denominator = len(values) * np.square(values).sum()
    return float(values.sum() ** 2 / denominator) if denominator else 1.0


def calculate_metrics(demand, allocation, capacity, previous_backlog, duration=1.0):
    demand = np.asarray(demand, dtype=float); allocation = np.asarray(allocation, dtype=float)
    served = np.minimum(demand + np.asarray(previous_backlog), allocation)
    backlog = np.maximum(demand + np.asarray(previous_backlog) - served, 0)
    remaining = backlog
    queue_cap = capacity * MAX_BACKLOG_CAPACITY_MULTIPLE / max(len(demand), 1)
    dropped = np.maximum(remaining - queue_cap, 0)
    backlog = np.minimum(remaining, queue_cap)
    allocation_floor = capacity * MIN_QUEUE_ALLOCATION_FRACTION / max(len(demand), 1)
    ratio = np.divide(backlog, np.maximum(allocation, allocation_floor),
                      out=np.zeros_like(backlog), where=backlog > 0)
    delay = np.minimum(BASE_LATENCY_MS + QUEUE_DELAY_SCALE_MS * np.minimum(ratio, 10.0), MAX_QUEUE_DELAY_MS)
    delay[(demand + previous_backlog > 0) & (allocation <= 0)] = MAX_QUEUE_DELAY_MS
    loss = np.divide(dropped, demand + np.asarray(previous_backlog), out=np.zeros_like(backlog), where=(demand + previous_backlog) > 0)
    latency = float(np.mean(delay)) if len(delay) else 0.0
    jitter = float(np.mean(np.abs(delay[1:] - delay[:-1]))) if len(delay) > 1 else 0.0
    return {"throughput": float(served.sum()), "latency": latency,
            "packet_loss": float(np.mean(loss) * 100) if len(loss) else 0.0,
            "jitter": jitter, "qos_satisfaction": float(np.mean((delay <= QOS_LATENCY_TARGET_MS) & (loss <= .01)) * 100) if len(delay) else 0.0,
            "jain_fairness": jain_fairness(served),
            "resource_utilization": float(allocation.sum() / capacity * 100) if capacity > 0 else 0.0,
            "backlog": backlog, "duration": duration}
