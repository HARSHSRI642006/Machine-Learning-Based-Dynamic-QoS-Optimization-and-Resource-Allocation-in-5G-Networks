import numpy as np
from .config import (ALPHA, BETA, GAMMA, PROPOSED_MIN_EQUAL_SHARE_FRACTION,
                     FAIRNESS_WEIGHT_MIN, FAIRNESS_WEIGHT_MAX)


def equal_share(demand, capacity):
    return _cap_allocation(np.full(len(demand), capacity / max(len(demand), 1)), demand, capacity)


def round_robin(demand, capacity):
    # Vectorized progressive fill is equivalent to repeated equal round-robin quanta:
    # capped streams drop out, and the remaining capacity is shared by the rest.
    return weighted_allocate(demand, capacity, np.ones(len(demand)))


def proportional_fair(demand, capacity, history=None):
    history = np.ones(len(demand)) if history is None else np.maximum(np.asarray(history), 1e-9)
    weights = 1 / history
    return weighted_allocate(demand, capacity, weights)


def predictive_allocation(predicted, capacity, priorities, urgency, fairness=True, priority=True, history=None):
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    demand_weight = predicted / max(float(predicted.max()), 1e-9)
    p = np.asarray(priorities, dtype=float) if priority else np.zeros(len(predicted))
    u = np.asarray(urgency, dtype=float)
    weights = ALPHA * demand_weight + (BETA if priority else 0) * p + GAMMA * u
    capacity = max(float(capacity), 0.0)
    if not fairness:
        return weighted_allocate(predicted, capacity, weights)

    # Bounded proportional-fair correction: historical service can alter a
    # stream's weight by at most 25% in either direction. A reserved minimum
    # equal-share grant prevents a nonzero-demand stream from being starved.
    active = predicted > 0
    if not active.any() or predicted.sum() <= capacity:
        return predicted.copy() if predicted.sum() <= capacity else np.zeros_like(predicted)
    if history is not None:
        history = np.maximum(np.asarray(history, dtype=float), 1e-9)
        correction = np.clip(np.mean(history) / history,
                             FAIRNESS_WEIGHT_MIN, FAIRNESS_WEIGHT_MAX)
        weights *= correction
    weights = np.maximum(weights, 1e-9)
    floor = np.zeros_like(predicted)
    floor[active] = np.minimum(predicted[active],
        capacity / max(int(active.sum()), 1) * PROPOSED_MIN_EQUAL_SHARE_FRACTION)
    residual_demand = np.maximum(predicted - floor, 0)
    residual = weighted_allocate(residual_demand, capacity - floor.sum(), weights)
    allocation = floor + residual
    return _cap_allocation(allocation, predicted, capacity)


def weighted_allocate(demand, capacity, weights):
    demand = np.maximum(np.asarray(demand, dtype=float), 0)
    weights = np.maximum(np.asarray(weights, dtype=float), 0)
    capacity = max(float(capacity), 0)
    if demand.sum() <= capacity:
        return demand
    positive = (demand > 0) & (weights > 0)
    allocation = np.zeros(len(demand))
    if not positive.any():
        positive = demand > 0
        weights[positive] = 1.0
    indexes = np.flatnonzero(positive)
    d, w = demand[indexes], weights[indexes]
    order = np.argsort(d / w)
    sorted_d, sorted_w = d[order], w[order]
    ratios = sorted_d / sorted_w
    cumulative_d = np.cumsum(sorted_d)
    suffix_w = np.cumsum(sorted_w[::-1])[::-1]
    after_w = np.r_[suffix_w[1:], 0.0]
    thresholds = cumulative_d + ratios * after_w
    boundary = int(np.searchsorted(thresholds, capacity, side="left"))
    if boundary == 0:
        level = capacity / sorted_w.sum()
    elif boundary >= len(sorted_d):
        level = np.inf
    else:
        level = (capacity - cumulative_d[boundary - 1]) / suffix_w[boundary]
    sorted_allocation = np.minimum(sorted_d, level * sorted_w)
    allocation[indexes[order]] = sorted_allocation
    return _cap_allocation(allocation, demand, capacity)


def _cap_allocation(values, demand, capacity):
    values = np.maximum(np.minimum(values, demand), 0)
    total = values.sum()
    if total > capacity and total > 0: values *= capacity / total
    return values
