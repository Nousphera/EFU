"""Plaintext sample-weighted aggregation used by the experiment coordinator."""

import torch
import numpy as np


def fedavg_aggr(dicts, num_samples, rnd, eta=0.001, model=None, **kwargs):
    """Sample-weight all floating state; integer counters use the maximum."""
    if not dicts or len(dicts) != len(num_samples):
        raise ValueError("Every state needs a positive sample count")
    if any(type(count) is not int or count <= 0 for count in num_samples):
        raise ValueError("Sample counts must be positive integers")
    layout = [(name, value.shape, value.dtype) for name, value in dicts[0].items()]
    if any(
        [(name, value.shape, value.dtype) for name, value in state.items()] != layout
        for state in dicts
    ):
        raise ValueError("Client state layouts differ")
    if rnd < 1:
        raise ValueError("Round numbers are one-based")
    weights = [count / sum(num_samples) for count in num_samples]
    result = {}
    for name, template in dicts[0].items():
        if template.is_floating_point():
            if any(not torch.isfinite(state[name]).all() for state in dicts):
                raise ValueError("Nonfinite client state")
            # Higher precision accumulation avoids cohort-size rounding drift.
            value = sum(
                state[name].detach().cpu().double() * weight
                for state, weight in zip(dicts, weights)
            )
            result[name] = value.to(template.dtype)
        else:
            result[name] = torch.stack(
                [state[name].detach().cpu() for state in dicts]
            ).amax(0)
    if model is not None:
        model.load_state_dict(result, strict=True)
    return result, None, None, eta / np.sqrt(rnd)
