"""Primary-equation unlearning kernels; FedOSD projection is cohort-side.

PGD: https://arxiv.org/abs/2207.05521, Algorithm 1.
FedOSD: https://arxiv.org/abs/2412.20200, Eqs. 3, 5-7.
SGA-EWC uses a declared empirical-Fisher estimate and a minimizing loss sign.
"""

import math
import torch
import torch.nn.functional as F
from .ewc import empirical_fisher, quadratic_penalty


def parameter_reference(model):
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def flatten_parameters(model):
    return torch.cat(
        [p.detach().reshape(-1) for p in model.parameters() if p.requires_grad]
    )


def set_parameter_vector(model, vector):
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                length = parameter.numel()
                parameter.copy_(vector[offset : offset + length].view_as(parameter))
                offset += length
    if offset != vector.numel():
        raise ValueError("Parameter vector has an incompatible length")


def leave_one_out_reference(global_state, contribution, weight):
    if not math.isfinite(weight) or not 0 < weight < 1:
        raise ValueError("Requester weight must lie strictly between zero and one")
    if global_state.keys() != contribution.keys():
        raise ValueError("Reference and contribution parameter layouts differ")
    return {
        n: (value.detach().cpu() - weight * contribution[n].detach().cpu())
        / (1 - weight)
        for n, value in global_state.items()
    }


def project_parameters(model, reference, radius):
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("PGD radius must be finite and positive")
    current = flatten_parameters(model)
    anchor = torch.cat(
        [
            reference[n].to(p).reshape(-1)
            for n, p in model.named_parameters()
            if p.requires_grad
        ]
    )
    delta = current - anchor
    norm = delta.norm()
    if norm > radius:
        set_parameter_vector(model, anchor + delta * (radius / norm))


def prepare_sga_ewc(model, fisher_data):
    return {
        "reference": parameter_reference(model),
        "fisher": empirical_fisher(model, fisher_data),
    }


def _labels(model, labels):
    return torch.tensor(
        [model.label2class[int(y)] for y in labels], device=model.device
    )


def pgd_halimi(
    model,
    data,
    num_epochs=5,
    lr=0.01,
    delta=3.0,
    *,
    prepared,
    lambda_mas=0.0,
    momentum=0.9,
    grad_clip=5.0,
    retain_objective=None,
):
    """Projected ascent around a fixed leave-one-out reference.

    EFU fixes the number of rounds; no hidden early stopping is applied. Radius
    is explicit (the original paper calibrated it from random models).
    """
    model.train()
    if not prepared.get("initialized", False):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(prepared["reference"][name].to(param))
        prepared["initialized"] = True
    project_parameters(model, prepared["reference"], delta)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    total, count = 0.0, 0
    for _ in range(num_epochs):
        for x, y in data:
            x, y = x.to(model.device), _labels(model, y)
            optimizer.zero_grad(set_to_none=True)
            loss = -F.cross_entropy(model(x), y)
            if lambda_mas:
                loss = loss + lambda_mas * quadratic_penalty(
                    model, prepared["importance"], prepared["mas_reference"]
                )
            if retain_objective is not None:
                loss = loss + retain_objective(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            project_parameters(model, prepared["reference"], delta)
            total += float(loss.detach()) * len(y)
            count += len(y)
    if not count:
        raise ValueError("Forget dataset must be nonempty")
    return model, len(data.dataset), {"loss": total / count}


def sga_ewc(
    model,
    data,
    num_epochs=5,
    lr=0.001,
    lambda_ewc=1.0,
    *,
    prepared=None,
    retain_objective=None,
):
    if prepared is None:
        prepared = prepare_sga_ewc(model, data)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    total, count = 0.0, 0
    for _ in range(num_epochs):
        for x, y in data:
            x, y = x.to(model.device), _labels(model, y)
            optimizer.zero_grad(set_to_none=True)
            loss = -F.cross_entropy(model(x), y) + 0.5 * lambda_ewc * quadratic_penalty(
                model, prepared["fisher"], prepared["reference"]
            )
            if retain_objective is not None:
                loss = loss + retain_objective(x)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
    if not count:
        raise ValueError("Forget dataset must be nonempty")
    return model, len(data.dataset), {"loss": total / count}


def unlearning_cross_entropy(logits, labels):
    probabilities = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
    return -torch.log1p(-0.5 * probabilities).mean()


def fedosd_direction(forget_gradient, retained_gradients, tolerance=1e-7):
    """Norm-matched negative nullspace projection; zero when none is available."""
    g, G = forget_gradient, retained_gradients.to(forget_gradient)
    if g.ndim != 1 or G.ndim != 2 or G.shape[1] != g.numel():
        raise ValueError("Incompatible FedOSD gradient shapes")
    if not torch.isfinite(g).all() or not torch.isfinite(G).all():
        raise ValueError("FedOSD gradients must be finite")
    work_g, work_G = g.double(), G.double()
    residual = (
        work_g
        if not len(G)
        else work_g
        - work_G.T
        @ (torch.linalg.pinv(work_G @ work_G.T, rtol=tolerance) @ (work_G @ work_g))
    )
    norm = residual.norm()
    if norm <= tolerance * max(float(work_g.norm()), 1.0):
        return torch.zeros_like(g)
    return (-residual * (work_g.norm() / norm)).to(g)


def project_recovery_gradient(gradient, displacement, tolerance=1e-7):
    """Algorithm 1: project only a gradient that moves toward the anchor."""
    a = displacement.to(gradient)
    norm2, dot = a.square().sum(), gradient.dot(a)
    if norm2 <= tolerance**2 or dot <= 0:
        return gradient.clone()
    projected = gradient - dot / norm2 * a
    norm = projected.norm()
    if norm <= tolerance * max(float(gradient.norm()), 1.0):
        return torch.zeros_like(gradient)
    return projected * (gradient.norm() / norm)


def fedosd(*args, **kwargs):
    raise ValueError(
        "FedOSD requires cohort gradients; use Experiment.aggregate_plaintext"
    )
