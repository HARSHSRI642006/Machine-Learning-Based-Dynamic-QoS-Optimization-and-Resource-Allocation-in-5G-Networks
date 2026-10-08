import numpy as np
from . import allocators
from .metrics import jain_fairness
from .config import (BASE_LATENCY_MS, MAX_BACKLOG_CAPACITY_MULTIPLE,
                     MIN_QUEUE_ALLOCATION_FRACTION, QUEUE_DELAY_SCALE_MS,
                     MAX_QUEUE_DELAY_MS, QOS_LATENCY_TARGET_MS)


def run_method(method, demand_series, capacity, priorities, seed, predictions=None,
               prediction_weight=0.0, ablation=None):
    demand_series = np.asarray(demand_series, dtype=float)
    users, steps = demand_series.shape
    backlog = np.zeros(users); delivered_total = 0.0; sent_total = 0.0; dropped_total = 0.0
    delays = []; losses = []; qos_scores = []; allocations = []
    historical = np.ones(users)
    for t in range(steps):
        demand = demand_series[:, t]
        prediction = demand if predictions is None else predictions[:, t]
        blended_demand = prediction_weight * prediction + (1.0 - prediction_weight) * demand
        urgency = np.clip(demand / max(np.mean(demand), 1e-9), 0, 2)
        if method == "Equal Share": allocation = allocators.equal_share(demand + backlog, capacity)
        elif method == "Round Robin": allocation = allocators.round_robin(demand + backlog, capacity)
        elif method == "Proportional Fair": allocation = allocators.proportional_fair(demand + backlog, capacity, historical)
        else:
            allocation = allocators.predictive_allocation(blended_demand + backlog, capacity, priorities, urgency,
                fairness=ablation != "NO FAIRNESS", priority=ablation != "NO PRIORITY", history=historical)
            if ablation == "SINGLE OBJECTIVE": allocation = allocators.weighted_allocate(blended_demand + backlog, capacity, np.maximum(blended_demand, 0))
        # One input row is one modeled observation; its physical duration and
        # bitrate units are undocumented, so values stay in simulation units.
        offered = demand + backlog
        served = np.minimum(offered, allocation)
        remaining = np.maximum(offered - served, 0)
        # Shared finite buffer (two capacity-seconds across the whole system).
        # Overflow is explicitly dropped; persistent overload cannot create an
        # unbounded queue. The per-stream cap keeps the model symmetric.
        queue_cap = capacity * MAX_BACKLOG_CAPACITY_MULTIPLE / max(users, 1)
        dropped = np.maximum(remaining - queue_cap, 0)
        backlog = np.minimum(remaining, queue_cap)
        allocation_floor = capacity * MIN_QUEUE_ALLOCATION_FRACTION / max(users, 1)
        queue_ratio = np.divide(backlog, np.maximum(allocation, allocation_floor),
                                out=np.zeros_like(backlog), where=(backlog > 0))
        queue_wait = QUEUE_DELAY_SCALE_MS * np.minimum(queue_ratio, 10.0)
        delay = np.minimum(BASE_LATENCY_MS + queue_wait, MAX_QUEUE_DELAY_MS)
        # An unserved nonempty stream is explicitly assigned the documented cap.
        delay[(offered > 0) & (allocation <= 0)] = MAX_QUEUE_DELAY_MS
        lost = np.divide(dropped, offered, out=np.zeros_like(backlog), where=offered > 0)
        delays.append(float(np.mean(delay)))
        losses.append(float(np.mean(lost) * 100))
        qos_scores.append(float(np.mean((delay <= QOS_LATENCY_TARGET_MS) & (lost <= .01)) * 100))
        delivered_total += float(served.sum()); sent_total += float(demand.sum()); dropped_total += float(dropped.sum())
        allocations.append(allocation)
        historical = .8 * historical + .2 * allocation
    rates = np.mean(allocations, axis=0)
    return {"throughput": delivered_total / steps, "latency": float(np.mean(delays)),
            "packet_loss": float(np.clip(100 * dropped_total / max(sent_total, 1e-9), 0, 100)),
            "jitter": float(np.mean(np.abs(np.diff(delays)))) if len(delays) > 1 else 0,
            "qos_satisfaction": float(np.mean(qos_scores)),
            "jain_fairness": jain_fairness(rates),
            "resource_utilization": float(np.mean(np.sum(allocations, axis=1)) / capacity * 100),
            "seed": seed}
