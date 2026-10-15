"""Paper section 5.2 task unlearning with explicit maintained recipe choices.

Joint identity/gender learning -> all-client identity Eq. (3) + gender CE ->
gender-only learning. The secure aggregation backend remains coordinate FE.
"""

import copy
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .task_data import IdentityGenderDataset, attach_task_labels


class DualHeadModel(torch.nn.Module):
    def __init__(self, name, identities, image=True, pretrained=False):
        super().__init__()
        if identities < 2:
            raise ValueError("At least two identity classes are required")
        if name == "tiny":
            dimension = 8
            layers = (
                [torch.nn.AdaptiveAvgPool2d((2, 2)), torch.nn.Flatten()]
                if image
                else []
            )
            self.backbone = torch.nn.Sequential(
                *layers, torch.nn.Linear(12 if image else 4, dimension), torch.nn.ReLU()
            )
        else:
            from .train import ImageModel

            self.backbone = ImageModel(name, identities, pretrained)
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


def prepare_task_unlearning(model, data, attack_steps=100):
    """Cache once at request start, keeping gender labels aligned with attack rows."""
    from .client import prepare_unlearning

    batches = [(x, labels) for x, labels in data]
    identity_data = [(x, labels[:, 0]) for x, labels in batches]
    prepared = prepare_unlearning(
        IdentityView(model), identity_data, attack_steps=attack_steps
    )
    prepared["batches"] = [
        {
            "image": x,
            "identity": identity,
            "gender": labels[:, 1].to(model.device),
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
    logits = model(batch["image"])
    identity_model = IdentityView(model)
    regularizer = sum(
        (
            prepared["importance"][name]
            * (parameter - prepared["reference"][name]).square()
        ).sum()
        for name, parameter in identity_model.named_parameters()
        if name in prepared["importance"]
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


def unlearn_task(model, prepared, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    totals = {name: 0.0 for name in ("retain", "forget", "adversarial", "regularizer")}
    count = 0
    for _ in range(epochs):
        for batch in prepared["batches"]:
            optimizer.zero_grad(set_to_none=True)
            parts = task_objective(model, batch, prepared)
            sum(parts.values()).backward()
            optimizer.step()
            size = len(batch["identity"])
            for name, value in parts.items():
                totals[name] += float(value.detach()) * size
            count += size
    return {name: value / count for name, value in totals.items()}


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


def run_task(args):
    from .train import load_data, loader
    from .dataset_io import manifest_digest
    from .aggregate import fedavg_aggr
    from .utils.fe import fe_secure_aggregate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    train, test, partitions, _, classes = load_data(args)
    if args.dataset == "synthetic":
        train = IdentityGenderDataset(train, (train.tensors[0][:, 3] > 0.5).long())
        test = IdentityGenderDataset(test, (test.tensors[0][:, 3] > 0.5).long())
        annotation_metadata = {
            "label_origin": "artificial binary labels on generated feature vectors; no human subjects",
            "gender_to_idx": {"synthetic_0": 0, "synthetic_1": 1},
        }
    else:
        train, test, annotation_metadata = attach_task_labels(
            train, test, args.data_root
        )
    model = DualHeadModel(
        args.model, classes, args.dataset != "synthetic", args.pretrained
    ).to(args.device)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    history, prepared = [], {}
    for round_id in range(1, args.rounds + 1):
        if args.method == "retrain":
            phase = "gender-only"
        elif (
            args.method == "efu"
            and args.unlearn_start
            <= round_id
            < args.unlearn_start + args.unlearn_rounds
        ):
            phase = "identity-unlearning"
        elif (
            args.method == "efu"
            and round_id >= args.unlearn_start + args.unlearn_rounds
        ):
            phase = "gender-only"
        else:
            phase = "joint-learning"
        states, sizes, objectives = [], [], {}
        for client_id, indices in enumerate(partitions):
            local = copy.deepcopy(model)
            data = loader(train, indices, args, shuffle=phase != "identity-unlearning")
            if phase == "identity-unlearning":
                if client_id not in prepared:
                    prepared[client_id] = prepare_task_unlearning(
                        local, data, attack_steps=args.attack_steps
                    )
                objectives[client_id] = unlearn_task(
                    local, prepared[client_id], args.unlearn_epochs, args.lr
                )
            else:
                objectives[client_id] = fit_tasks(
                    local,
                    data,
                    args.local_epochs,
                    args.lr,
                    retain_only=phase == "gender-only",
                )
            states.append(
                {
                    name: value.detach().cpu().clone()
                    for name, value in local.state_dict().items()
                }
            )
            sizes.append(len(indices))
        updated = (
            fe_secure_aggregate(
                states,
                sizes,
                kappa=args.kappa,
                precision=args.precision,
                max_abs=args.max_abs,
                seed=args.seed,
            )
            if args.method == "efu"
            else fedavg_aggr(states, sizes, rnd=round_id)[0]
        )
        full = model.state_dict()
        full.update(
            {
                name: torch.as_tensor(value, dtype=full[name].dtype, device=args.device)
                for name, value in updated.items()
            }
        )
        model.load_state_dict(full, strict=True)
        row = {
            "round": round_id,
            "phase": phase,
            "participants": list(range(args.clients)),
            "unlearning_clients": (
                list(range(args.clients)) if phase == "identity-unlearning" else []
            ),
            "sample_counts": sizes,
            "local_objectives": objectives,
            "test": evaluate_tasks(model, test_loader),
        }
        history.append(row)
        print(json.dumps(row))
    result = {
        "project": "EFU",
        "configuration": vars(args),
        "partitions": partitions,
        "history": history,
        "protocol": (
            "decentralized-coordinate-fe"
            if args.method == "efu"
            else "plaintext-baseline"
        ),
        "benchmark_reproduction": False,
        "data_manifest_sha256": manifest_digest(args.dataset, args.data_root),
        "task_annotations": annotation_metadata,
        "task_recipe": {
            "source": "https://arxiv.org/html/2508.07873v1#S5.SS2",
            "round_indexing": "one-based; unlearn_start is the first unlearning round",
            "request_scope": "all clients, all local samples; forget-client/fraction are single-task options",
            "loss_weights": {
                "retain_gender": 1.0,
                "forget_identity": 1.0,
                "adversarial_identity": 1.0,
                "inverted_mas": 1.0,
            },
            "optimizer": "Adam, weight_decay=0.01; paper section 5.1",
            "mas_scope": "shared backbone and identity head; retained gender head excluded",
            "attack": "targeted L2 APGD, eps=0.4, random incorrect identity, cached at request start per client",
            "post_request": "gender-only learning; no identity supervision",
            "baseline": "retrain starts from initialization using only gender; fedavg continues joint learning",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        torch.save(
            {
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "configuration": vars(args),
                "classes": {"identity": classes, "gender": 2},
                "task_annotations": annotation_metadata,
                "task_recipe": result["task_recipe"],
            },
            path.with_suffix(".pt"),
        )
    return result
