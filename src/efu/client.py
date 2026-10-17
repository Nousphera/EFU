# Modified 2026-10-15: cache offline attacks and initial MAS/reference across unlearning updates.
import copy
import torch
from efu.utils.apgd import apgd
from efu.utils.mas import get_weights_importance


def prepare_unlearning(model, data, eps=0.4, attack_steps=100):
    """Prepare Eq. (3)'s fixed reference, inverted MAS and offline attacks.

    Inputs must be raw [0,1] pixels/features; put any normalization inside the
    model so APGD's projection and the model operate in consistent coordinates.
    Reuse the returned preparation for every epoch/round of the same request.
    """
    adv_model = copy.deepcopy(model).eval()
    num_classes = len(model.label2class)
    if num_classes < 2:
        raise ValueError("Unlearning requires at least two output classes")
    batches = []
    for x, y in data:
        x = x.to(model.device)
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(
                "APGD expects [0,1] inputs; put normalization inside the model"
            )
        y = torch.tensor(
            [model.label2class[int(label)] for label in y], device=model.device
        )
        offsets = torch.randint(1, num_classes, y.shape, device=model.device)
        adv_y = (y + offsets) % num_classes
        adv_x = apgd(
            model=adv_model,
            inputs=x,
            labels=adv_y,
            eps=eps,
            norm=2,
            targeted=True,
            n_iter=attack_steps,
            n_restarts=1,
            loss_function="ce",
            best_loss=True,
        )
        batches.append(tuple(value.detach().cpu() for value in (x, y, adv_x, adv_y)))
    if not batches:
        raise ValueError("Forget dataset must be nonempty")
    return {
        "batches": batches,
        "importance": get_weights_importance(model, data),
        "reference": {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        },
    }


def local_unlearn(
    model,
    data,
    num_epochs=1,
    lr=1e-3,
    lambda_imp=1.0,
    eps=0.4,
    prepared=None,
    attack_steps=100,
):
    """Optimize negative CE + offline targeted-adversarial CE + inverted MAS."""
    if prepared is None:
        prepared = prepare_unlearning(model, data, eps=eps, attack_steps=attack_steps)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    for _ in range(num_epochs):
        total_loss, correct, count = 0.0, 0, 0
        for x, y, adv_x, adv_y in prepared["batches"]:
            x, y, adv_x, adv_y = (
                value.to(model.device) for value in (x, y, adv_x, adv_y)
            )
            optimizer.zero_grad()
            logits = model(x)
            forget = -torch.nn.functional.cross_entropy(logits, y)
            adversarial = torch.nn.functional.cross_entropy(model(adv_x), adv_y)
            from .utils.ewc import quadratic_penalty

            regularizer = quadratic_penalty(
                model, prepared["importance"], prepared["reference"]
            )
            objective = forget + adversarial + lambda_imp * regularizer
            objective.backward()
            optimizer.step()
            total_loss += float(objective.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
    return (
        model,
        len(data.dataset),
        {"loss": total_loss / count, "accuracy": correct / count},
    )
