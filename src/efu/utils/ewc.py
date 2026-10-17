"""Diagonal empirical Fisher, distinct from squared minibatch gradients."""

import torch
from .mas import per_example_importance


def empirical_fisher(model, data):
    def score(logits, y):
        label = model.label2class[int(y)] if hasattr(model, "label2class") else int(y)
        return torch.log_softmax(logits, dim=-1)[0, label]

    return per_example_importance(model, data, score, 2)


def quadratic_penalty(model, importance, reference):
    terms = [
        (importance[n].to(p) * (p - reference[n].to(p)).square()).sum()
        for n, p in model.named_parameters()
        if p.requires_grad and n in importance
    ]
    return sum(terms, next(model.parameters()).new_zeros(()))
