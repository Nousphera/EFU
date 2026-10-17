"""Identity forgetting with retained gender learning (paper section 5.2).

Joint identity/gender learning -> all-client identity Eq. (3) + gender CE ->
gender-only learning. Transport and aggregation are handled by Experiment/Flower.
"""

import torch


class DualHeadModel(torch.nn.Module):
    def __init__(self, name, identities, pretrained=False, model_root="models"):
        super().__init__()
        if identities < 2:
            raise ValueError("At least two identity classes are required")
        from .train import ImageModel

        self.backbone = ImageModel(name, identities, pretrained, model_root=model_root)
        if name == "resnet18":
            dimension = self.backbone.body.fc.in_features
            self.backbone.body.fc = torch.nn.Identity()
        else:
            dimension = self.backbone.body.heads.head.in_features
            self.backbone.body.heads.head = torch.nn.Identity()
        self.identity_head = torch.nn.Linear(dimension, identities)
        self.gender_head = torch.nn.Linear(dimension, 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        features = self.backbone(x)
        return {
            "identity": self.identity_head(features),
            "gender": self.gender_head(features),
        }


class IdentityView(torch.nn.Module):
    """Route attacks and identity importance through shared backbone + identity head."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.identity_head
        self.label2class = {i: i for i in range(model.identity_head.out_features)}

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.head(self.backbone(x))


def fit_tasks(model, data, epochs, lr, retain_only=False):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    total, count = 0.0, 0
    for _ in range(epochs):
        for x, labels in data:
            x, labels = x.to(model.device), labels.to(model.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            objective = torch.nn.functional.cross_entropy(
                logits["gender"], labels[:, 1]
            )
            if not retain_only:
                objective = objective + torch.nn.functional.cross_entropy(
                    logits["identity"], labels[:, 0]
                )
            objective.backward()
            optimizer.step()
            total += float(objective.detach()) * len(x)
            count += len(x)
    return {"objective": total / count}


def prepare_task_unlearning(model, data, attack_steps=100, eps=0.4):
    """Cache once at request start, keeping gender labels aligned with attack rows."""
    from .client import prepare_unlearning

    batches = [(x, labels) for x, labels in data]
    identity_data = [(x, labels[:, 0]) for x, labels in batches]
    prepared = prepare_unlearning(
        IdentityView(model), identity_data, attack_steps=attack_steps, eps=eps
    )
    prepared["batches"] = [
        {
            "image": x,
            "identity": identity,
            "gender": labels[:, 1].cpu(),
            "adversarial": adversarial,
            "adversarial_identity": adversarial_identity,
        }
        for (x, identity, adversarial, adversarial_identity), (_, labels) in zip(
            prepared["batches"], batches
        )
    ]
    return prepared


def task_objective(model, batch, prepared):
    """Unit-weight simultaneous CE_gender + identity Eq. (3), not alternating steps."""
    batch = {name: value.to(model.device) for name, value in batch.items()}
    logits = model(batch["image"])
    identity_model = IdentityView(model)
    from .utils.ewc import quadratic_penalty

    regularizer = quadratic_penalty(
        identity_model, prepared["importance"], prepared["reference"]
    )
    return {
        "retain": torch.nn.functional.cross_entropy(logits["gender"], batch["gender"]),
        "forget": -torch.nn.functional.cross_entropy(
            logits["identity"], batch["identity"]
        ),
        "adversarial": torch.nn.functional.cross_entropy(
            identity_model(batch["adversarial"]), batch["adversarial_identity"]
        ),
        "regularizer": regularizer,
    }


def unlearn_task(model, prepared, epochs, lr, mas_weight=1.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    totals = {name: 0.0 for name in ("retain", "forget", "adversarial", "regularizer")}
    count = 0
    for _ in range(epochs):
        for batch in prepared["batches"]:
            optimizer.zero_grad(set_to_none=True)
            parts = task_objective(model, batch, prepared)
            parts["regularizer"] = mas_weight * parts["regularizer"]
            sum(parts.values()).backward()
            optimizer.step()
            size = len(batch["identity"])
            for name, value in parts.items():
                totals[name] += float(value.detach()) * size
            count += size
    return {name: value / count for name, value in totals.items()}


def prepare_task_baseline(model, data, method, *, reference=None, mas=False):
    """Explicit task adaptation: identity regularizer, simultaneous gender CE."""
    from .utils.baselines import parameter_reference, prepare_sga_ewc
    from .utils.mas import get_weights_importance

    view = IdentityView(model)
    batches = [(x.cpu(), y.cpu()) for x, y in data]
    identity_data = [(x, y[:, 0]) for x, y in batches]
    if method == "sga_ewc":
        result = prepare_sga_ewc(view, identity_data)
    else:
        if method == "pgd" and reference is None:
            raise ValueError("Task PGD requires a leave-one-out identity reference")
        result = {
            "reference": reference
            if reference is not None
            else parameter_reference(view)
        }
    if mas:
        result["importance"] = get_weights_importance(view, identity_data)
        result["mas_reference"] = parameter_reference(view)
    result["batches"] = batches
    return result


def unlearn_task_baseline(
    model, prepared, method, epochs, lr, *, mas_weight=0.0, ewc_weight=1.0, radius=3.0
):
    from .utils.baselines import project_parameters
    from .utils.ewc import quadratic_penalty

    view = IdentityView(model)
    model.train()
    if method == "pgd" and not prepared.get("initialized", False):
        with torch.no_grad():
            for name, parameter in view.named_parameters():
                parameter.copy_(prepared["reference"][name].to(parameter))
        prepared["initialized"] = True
    if method == "pgd":
        project_parameters(view, prepared["reference"], radius)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9 if method == "pgd" else 0.0
    )
    total, count = 0.0, 0
    for _ in range(epochs):
        for x, labels in prepared["batches"]:
            x, labels = x.to(model.device), labels.to(model.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            objective = torch.nn.functional.cross_entropy(
                outputs["gender"], labels[:, 1]
            ) - torch.nn.functional.cross_entropy(outputs["identity"], labels[:, 0])
            if method == "sga_ewc":
                objective = objective + 0.5 * ewc_weight * quadratic_penalty(
                    view, prepared["fisher"], prepared["reference"]
                )
            elif method == "pgd":
                if mas_weight:
                    objective = objective + mas_weight * quadratic_penalty(
                        view, prepared["importance"], prepared["mas_reference"]
                    )
            else:
                raise ValueError("Task baseline must be pgd or sga_ewc")
            objective.backward()
            if method == "pgd":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if method == "pgd":
                project_parameters(view, prepared["reference"], radius)
            total += float(objective.detach()) * len(x)
            count += len(x)
    return {"objective": total / count}


def evaluate_tasks(model, data):
    model.eval()
    totals = {name: {"correct": 0, "loss": 0.0} for name in ("identity", "gender")}
    count = 0
    with torch.no_grad():
        for x, labels in data:
            x, labels = x.to(model.device), labels.to(model.device)
            logits = model(x)
            for column, name in enumerate(("identity", "gender")):
                totals[name]["correct"] += int(
                    (logits[name].argmax(1) == labels[:, column]).sum()
                )
                totals[name]["loss"] += float(
                    torch.nn.functional.cross_entropy(
                        logits[name], labels[:, column], reduction="sum"
                    )
                )
            count += len(x)
    if count == 0:
        raise ValueError("Task evaluation split is empty")
    return {
        name: {
            "accuracy": values["correct"] / count,
            "loss": values["loss"] / count,
            "samples": count,
        }
        for name, values in totals.items()
    }
