"""Per-example MAS (EFU Eq. 1), with explicit normalization for inversion."""

import torch


def per_example_importance(model, data, scalar, power):
    """Accumulate before batching cancels gradients, preserving caller state."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not params:
        raise ValueError("Importance estimation requires trainable parameters")
    modes = {module: module.training for module in model.modules()}
    device = next(iter(params.values())).device
    result = {n: torch.zeros_like(p, device="cpu") for n, p in params.items()}
    count = 0
    try:
        model.eval()
        with torch.enable_grad():
            for xs, ys in data:
                for x, y in zip(xs, ys):
                    value = scalar(model(x.unsqueeze(0).to(device)), y.to(device))
                    grads = torch.autograd.grad(
                        value, tuple(params.values()), allow_unused=True
                    )
                    for (name, _), grad in zip(params.items(), grads):
                        if grad is not None:
                            result[name].add_(grad.detach().abs().pow(power).cpu())
                    count += 1
    finally:
        for module, training in modes.items():
            module.training = training
    if not count:
        raise ValueError("Importance dataset must be nonempty")
    return {name: value / count for name, value in result.items()}


def compute_mas_importance(model, data):
    """Mean absolute derivative of squared output norm, using model logits."""
    return per_example_importance(
        model, data, lambda output, _: output.square().sum(), 1
    )


def get_weights_importance(model, data):
    """Invert per-tensor min/max-normalized MAS; constant tensors receive one.

    Per-tensor normalization is an explicit reconstruction convention: EFU
    specifies inversion but does not provide a normalization formula.
    """
    importance = compute_mas_importance(model, data)
    for name, value in importance.items():
        lo, hi = value.min(), value.max()
        normalized = (value - lo) / (hi - lo) if hi > lo else torch.zeros_like(value)
        importance[name] = 1 - normalized
    return importance
