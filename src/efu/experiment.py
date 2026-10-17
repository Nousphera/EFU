"""Scientific experiment logic shared by the local runner and Flower transport.

Client caches contain only CPU tensors and weights_only-compatible containers.
No cache contains a model, loader, optimizer, or cryptographic secret object.
"""

import copy
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .requests import build_requests
from .utils.baselines import (
    parameter_reference,
    flatten_parameters,
    set_parameter_vector,
    leave_one_out_reference,
    prepare_sga_ewc,
    pgd_halimi,
    sga_ewc,
    unlearning_cross_entropy,
    fedosd_direction,
    project_recovery_gradient,
)
from .utils.ewc import quadratic_penalty
from .utils.mas import get_weights_importance


def cpu_state(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def observe_contribution(cache, state, round_id, weight):
    """Record the actual transmitted state (quantized reconstruction for FE)."""
    cache["last_contribution"] = {
        n: value.detach().cpu().clone() for n, value in state.items()
    }
    cache["last_round"] = int(round_id)
    cache["last_weight"] = float(weight)


def evaluate_class_splits(model, data, forget_classes):
    totals = {
        name: [0, 0.0, 0] for name in ("all_test", "retained_test", "forgotten_test")
    }
    model.eval()
    with torch.no_grad():
        for x, labels in data:
            x, labels = x.to(model.device), labels.to(model.device)
            logits = model(x)
            losses = F.cross_entropy(logits, labels, reduction="none")
            correct = logits.argmax(1) == labels
            forgotten = torch.zeros_like(labels, dtype=torch.bool)
            for label in forget_classes:
                forgotten |= labels == label
            masks = {
                "all_test": torch.ones_like(forgotten),
                "retained_test": ~forgotten,
                "forgotten_test": forgotten,
            }
            for name, mask in masks.items():
                totals[name][0] += int(correct[mask].sum())
                totals[name][1] += float(losses[mask].sum())
                totals[name][2] += int(mask.sum())
    return {
        name: {
            "accuracy": correct / count if count else None,
            "loss": loss / count if count else None,
            "samples": count,
        }
        for name, (correct, loss, count) in totals.items()
    }


class Experiment:
    observe_contribution = staticmethod(observe_contribution)

    def __init__(self, args):
        from .train import load_data, ImageModel, validate_args
        from .dataset_io import manifest_digest
        from .task_unlearning import DualHeadModel
        from .task_data import attach_task_labels

        self.args = validate_args(args)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.train_data, self.test_data, self.partitions, targets, self.classes = (
            load_data(args)
        )
        self.task = args.task == "identity-gender"
        self.annotation_metadata = None
        if self.task:
            self.train_data, self.test_data, self.annotation_metadata = (
                attach_task_labels(self.train_data, self.test_data, args.data_root)
            )
            self.model = DualHeadModel(
                args.model, self.classes, args.pretrained, args.model_root
            ).to(args.device)
            self.forget_by_client = {
                cid: list(indices) for cid, indices in enumerate(self.partitions)
            }
            self.retain_by_client = copy.deepcopy(self.partitions)
            self.forget_classes = []
        else:
            self.model = ImageModel(
                args.model, self.classes, args.pretrained, args.model_root
            ).to(args.device)
            requesters = (
                args.forget_clients
                if args.forget_clients is not None
                else [args.forget_client]
            )
            classes = (
                args.forget_classes
                if args.forget_classes is not None
                else ([args.forget_class] if args.forget_class is not None else None)
            )
            self.forget_by_client, self.retain_by_client, self.forget_classes = (
                build_requests(
                    self.partitions,
                    targets,
                    requesters,
                    args.forget_fraction,
                    seed=args.request_seed,
                    forget_classes=classes,
                    class_fraction=args.forget_class_fraction,
                )
            )
        self.initial_model_sha256 = state_digest(cpu_state(self.model))
        self.manifest_sha256 = manifest_digest(args.dataset, args.data_root)
        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size)
        self.request_reference = None

    def phase(self, round_id):
        args = self.args
        if args.method == "retrain":
            return "gender-only" if self.task else "retain-learning"
        if (
            args.method == "efu"
            and args.unlearn_start
            <= round_id
            < args.unlearn_start + args.unlearn_rounds
        ):
            return "identity-unlearning" if self.task else "unlearning"
        if (
            args.method == "efu"
            and round_id >= args.unlearn_start + args.unlearn_rounds
        ):
            return "gender-only" if self.task else "retain-learning"
        return "joint-learning" if self.task else "learning"

    def is_unlearning(self, round_id):
        return self.phase(round_id) in ("unlearning", "identity-unlearning")

    def participants(self, round_id):
        phase, args = self.phase(round_id), self.args
        active = [
            cid
            for cid in range(args.clients)
            if self.task or phase != "retain-learning" or self.retain_by_client[cid]
        ]
        count = math.floor(len(active) * args.participation)
        if count < 2:
            raise ValueError(
                "At least two retained active clients must remain; choose a different cohort"
            )
        required = []
        if args.method == "efu":
            # PGD must see the exact immediately preceding aggregation, then
            # initialize its fixed reference on the first unlearning round.
            pgd_preparation = (
                args.unlearn_method == "pgd" and round_id == args.unlearn_start - 1
            )
            mandatory = self.is_unlearning(round_id) and (
                args.request_participation == "required" or args.unlearn_method == "pgd"
            )
            if pgd_preparation or mandatory:
                required = sorted(self.forget_by_client)
        if len(required) > count:
            raise ValueError(
                "Requested participation cannot include every mandatory requester"
            )
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, round_id]))
        others = [cid for cid in active if cid not in required]
        return sorted(
            required + rng.choice(others, count - len(required), replace=False).tolist()
        )

    def sample_count(self, cid, round_id):
        if not self.task and self.phase(round_id) == "retain-learning":
            return len(self.retain_by_client[cid])
        return len(self.partitions[cid])

    def _loader(self, indices, *, shuffle=True):
        from .train import loader

        return loader(self.train_data, indices, self.args, shuffle)

    def _identity_reference(self, full):
        return {
            (
                "head." + name[len("identity_head.") :]
                if name.startswith("identity_head.")
                else name
            ): value
            for name, value in full.items()
            if name.startswith(("backbone.", "identity_head."))
        }

    def _pgd_reference(self, model, global_state, round_id, cache):
        if cache.get("last_round") != round_id - 1 or "last_contribution" not in cache:
            raise ValueError(
                "PGD needs the requester contribution from the immediately preceding preparation cohort"
            )
        names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        reference = leave_one_out_reference(
            {n: global_state[n] for n in names},
            {n: cache["last_contribution"][n] for n in names},
            cache["last_weight"],
        )
        return self._identity_reference(reference) if self.task else reference

    def _fit_fedosd(
        self, model, data, *, forget=False, prepared=None, retained_task=False
    ):
        args = self.args
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        epochs = args.unlearn_epochs if forget else args.local_epochs
        total, count = 0.0, 0
        for _ in range(epochs):
            for x, y in data:
                x, y = x.to(model.device), y.to(model.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                if self.task:
                    logits, labels = (
                        (output["gender"], y[:, 1])
                        if retained_task
                        else (output["identity"], y[:, 0])
                    )
                else:
                    logits, labels = output, y
                objective = (
                    unlearning_cross_entropy(logits, labels)
                    if forget
                    else F.cross_entropy(logits, labels)
                )
                if forget and prepared and "importance" in prepared:
                    from .task_unlearning import IdentityView

                    view = IdentityView(model) if self.task else model
                    objective = objective + args.mas_weight * quadratic_penalty(
                        view, prepared["importance"], prepared["reference"]
                    )
                objective.backward()
                optimizer.step()
                total += float(objective.detach()) * len(y)
                count += len(y)
        return {"objective": total / count}

    def client_update(
        self, cid, global_state, round_id, cache, *, request_reference=None
    ):
        from .train import fit
        from . import task_unlearning as task
        from .client import prepare_unlearning, local_unlearn

        args, phase = self.args, self.phase(round_id)
        if cid not in self.participants(round_id):
            raise ValueError("Client is not in this round cohort")
        # Independent client RNGs keep local/Flower runs equivalent under reordering.
        seed = int(
            np.random.SeedSequence([args.seed, round_id, cid]).generate_state(1)[0]
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        local = copy.deepcopy(self.model)
        local.load_state_dict(global_state, strict=True)
        unlearn = self.is_unlearning(round_id) and cid in self.forget_by_client
        mas = args.variant == "efu" and args.mas_weight > 0
        if request_reference is not None:
            cache["request_reference"] = {
                name: request_reference[name].detach().cpu().clone()
                for name, p in local.named_parameters()
                if p.requires_grad
            }
        if (
            args.unlearn_method == "fedosd"
            and round_id == args.unlearn_start
            and "request_reference" not in cache
        ):
            cache["request_reference"] = parameter_reference(local)
        prepared = cache.get("prepared")
        preparation_start = time.perf_counter()
        if unlearn and prepared is None:
            data = self._loader(self.forget_by_client[cid], shuffle=False)
            if args.unlearn_method == "efu":
                prepared = (
                    task.prepare_task_unlearning(
                        local, data, attack_steps=args.attack_steps, eps=args.attack_eps
                    )
                    if self.task
                    else prepare_unlearning(
                        local, data, attack_steps=args.attack_steps, eps=args.attack_eps
                    )
                )
            elif args.unlearn_method == "pgd":
                reference = self._pgd_reference(local, global_state, round_id, cache)
                if self.task:
                    prepared = task.prepare_task_baseline(
                        local, data, "pgd", reference=reference, mas=mas
                    )
                else:
                    prepared = {"reference": reference}
                    if mas:
                        prepared.update(
                            importance=get_weights_importance(local, data),
                            mas_reference=parameter_reference(local),
                        )
            elif args.unlearn_method == "sga_ewc":
                if self.task:
                    if args.fisher_source != "forget":
                        raise ValueError(
                            "Task SGA-EWC estimates identity Fisher; select fisher-source=forget"
                        )
                    prepared = task.prepare_task_baseline(local, data, "sga_ewc")
                else:
                    fisher_data = (
                        data
                        if args.fisher_source == "forget"
                        else self._loader(self.retain_by_client[cid], shuffle=False)
                    )
                    prepared = prepare_sga_ewc(local, fisher_data)
            else:
                view = task.IdentityView(local) if self.task else local
                prepared = {"reference": parameter_reference(view)}
                if mas:
                    identity_data = (
                        [(x, y[:, 0]) for x, y in data] if self.task else data
                    )
                    prepared["importance"] = get_weights_importance(view, identity_data)
            cache["prepared"] = prepared
        preparation_seconds = time.perf_counter() - preparation_start
        train_start = time.perf_counter()
        if unlearn:
            data = self._loader(self.forget_by_client[cid])
            if args.unlearn_method == "efu":
                if self.task:
                    metrics = task.unlearn_task(
                        local, prepared, args.unlearn_epochs, args.lr, args.mas_weight
                    )
                else:
                    _, _, metrics = local_unlearn(
                        local,
                        data,
                        args.unlearn_epochs,
                        args.lr,
                        lambda_imp=args.mas_weight,
                        prepared=prepared,
                    )
            elif args.unlearn_method in ("pgd", "sga_ewc"):
                if self.task:
                    metrics = task.unlearn_task_baseline(
                        local,
                        prepared,
                        args.unlearn_method,
                        args.unlearn_epochs,
                        args.lr,
                        mas_weight=args.mas_weight if mas else 0.0,
                        ewc_weight=args.ewc_weight,
                        radius=args.pgd_radius,
                    )
                elif args.unlearn_method == "pgd":
                    _, _, metrics = pgd_halimi(
                        local,
                        data,
                        args.unlearn_epochs,
                        args.lr,
                        args.pgd_radius,
                        prepared=prepared,
                        lambda_mas=args.mas_weight if mas else 0.0,
                    )
                else:
                    _, _, metrics = sga_ewc(
                        local,
                        data,
                        args.unlearn_epochs,
                        args.lr,
                        lambda_ewc=args.ewc_weight,
                        prepared=prepared,
                    )
            else:
                metrics = self._fit_fedosd(local, data, forget=True, prepared=prepared)
                if self.task:
                    retained_model = copy.deepcopy(self.model)
                    retained_model.load_state_dict(global_state, strict=True)
                    origin = flatten_parameters(retained_model)
                    self._fit_fedosd(
                        retained_model,
                        self._loader(self.partitions[cid]),
                        retained_task=True,
                    )
                    metrics["_retained_gradient"] = (
                        (origin - flatten_parameters(retained_model)) / args.lr
                    ).cpu()
        else:
            indices = (
                self.retain_by_client[cid]
                if phase in ("retain-learning", "gender-only")
                else self.partitions[cid]
            )
            if not indices:
                raise ValueError(
                    "Empty retained client must be omitted from the cohort"
                )
            data = self._loader(indices)
            if (
                args.method == "efu"
                and args.unlearn_method == "fedosd"
                and round_id >= args.unlearn_start
            ):
                origin = flatten_parameters(local)
                metrics = self._fit_fedosd(local, data, retained_task=self.task)
                if phase in ("retain-learning", "gender-only"):
                    if "request_reference" not in cache:
                        raise ValueError(
                            "FedOSD recovery needs the frozen pre-request global reference"
                        )
                    anchor = torch.cat(
                        [
                            cache["request_reference"][n].to(p).flatten()
                            for n, p in local.named_parameters()
                            if p.requires_grad
                        ]
                    )
                    gradient = (origin - flatten_parameters(local)) / args.lr
                    projected = project_recovery_gradient(gradient, origin - anchor)
                    set_parameter_vector(local, origin - args.lr * projected)
            elif self.task:
                metrics = task.fit_tasks(
                    local,
                    data,
                    args.local_epochs,
                    args.lr,
                    retain_only=phase == "gender-only",
                )
            else:
                fit(local, data, args.local_epochs, args.lr)
                metrics = {}
        metrics.update(
            training_seconds=time.perf_counter() - train_start,
            preparation_seconds=preparation_seconds,
        )
        state = cpu_state(local)
        size = self.sample_count(cid, round_id)
        counts = [self.sample_count(i, round_id) for i in self.participants(round_id)]
        # Only PGD needs historical client models; do not keep redundant full
        # model copies for EFU/Fisher/FedOSD clients.
        if args.unlearn_method == "pgd" and round_id == args.unlearn_start - 1:
            observe_contribution(cache, state, round_id, size / sum(counts))
        return state, size, metrics

    def aggregate_plaintext(
        self, states, counts, client_ids, global_state, round_id, *, metrics=None
    ):
        from .aggregate import fedavg_aggr

        if (
            len(states) != len(client_ids)
            or len(states) != len(counts)
            or len(set(client_ids)) != len(client_ids)
            or sorted(client_ids) != self.participants(round_id)
        ):
            raise ValueError("Expected the exact unique client cohort for this round")
        result = fedavg_aggr(states, counts, round_id)[0]
        if (
            self.args.method != "efu"
            or self.args.unlearn_method != "fedosd"
            or not self.is_unlearning(round_id)
        ):
            return result
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        origin = torch.cat([global_state[n].flatten().cpu() for n in names])
        gradients = [
            (origin - torch.cat([state[n].flatten().cpu() for n in names]))
            / self.args.lr
            for state in states
        ]
        forgetting = [
            g for cid, g in zip(client_ids, gradients) if cid in self.forget_by_client
        ]
        retained = [
            g
            for cid, g in zip(client_ids, gradients)
            if cid not in self.forget_by_client
        ]
        if not forgetting:
            return result
        if self.task:
            if metrics is None or len(metrics) != len(states):
                raise ValueError(
                    "Task FedOSD needs a retained gender gradient from each client"
                )
            for values in metrics:
                gradient = values.get("_retained_gradient")
                if (
                    not isinstance(gradient, torch.Tensor)
                    or gradient.shape != origin.shape
                    or not torch.isfinite(gradient).all()
                ):
                    raise ValueError("Invalid task retained gradient")
                retained.append(gradient.cpu())
        if not retained:
            raise ValueError("FedOSD needs at least one retained objective gradient")
        direction = fedosd_direction(
            torch.stack(forgetting).mean(0), torch.stack(retained)
        )
        vector = origin + self.args.lr * direction
        offset = 0
        for name in names:
            length = result[name].numel()
            result[name] = vector[offset : offset + length].view_as(result[name])
            offset += length
        return result

    def evaluate(self, state):
        from .train import evaluate
        from .task_unlearning import evaluate_tasks

        self.model.load_state_dict(state, strict=True)
        if self.task:
            return {"test": evaluate_tasks(self.model, self.test_loader)}
        result = evaluate_class_splits(
            self.model, self.test_loader, self.forget_classes
        )
        result.update(
            accuracy=result["all_test"]["accuracy"], loss=result["all_test"]["loss"]
        )
        indices = sorted(i for values in self.forget_by_client.values() for i in values)
        result["forget_train"] = evaluate(
            self.model, self._loader(indices, shuffle=False)
        )
        result["forget_train_accuracy"] = result["forget_train"]["accuracy"]
        return result

    def result(self, history):
        args = self.args
        result = {
            "project": "EFU",
            "configuration": vars(args),
            "partitions": self.partitions,
            "forget_indices_by_client": self.forget_by_client,
            "forget_classes": self.forget_classes,
            "history": history,
            "benchmark_reproduction": False,
            "protocol": "encrypted-centroid-fe"
            if args.method == "efu" and args.aggregation == "encrypted"
            else "plaintext-baseline",
            "initial_model_sha256": self.initial_model_sha256,
            "data_manifest_sha256": self.manifest_sha256,
            "method_recipe": {
                "mas": "per-example absolute output-norm gradient; per-tensor min/max inversion",
                "fisher": f"per-example empirical Fisher on {args.fisher_source} data; fixed request anchor",
                "pgd": "weighted leave-one-out reference; mandatory preparation cohort; explicit fixed radius; fixed rounds",
                "fedosd": "cohort nullspace projection and sign-conditional recovery; task route uses retained gender objectives",
                "optimizer": "Adam with weight_decay=0.01 for EFU; SGD for cited baseline updates",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
            },
        }
        if len(self.forget_by_client) == 1:
            result["forget_indices"] = next(iter(self.forget_by_client.values()))
        if self.task:
            result["task_annotations"] = self.annotation_metadata
            result["task_recipe"] = {
                "request_scope": "all clients; identity forgetting with retained gender",
                "loss_weights": {
                    "retain_gender": 1.0,
                    "forget_identity": 1.0,
                    "adversarial_identity": 1.0,
                    "inverted_mas": args.mas_weight,
                },
                "fedosd_adaptation": "retained gender gradients substitute retained-client objectives",
            }
        return result

    def run(self):
        args, history, caches = (
            self.args,
            [],
            {cid: {} for cid in range(self.args.clients)},
        )
        state = cpu_state(self.model)
        for round_id in range(1, args.rounds + 1):
            if (
                args.method == "efu"
                and args.unlearn_method == "fedosd"
                and round_id == args.unlearn_start
            ):
                self.request_reference = {
                    n: state[n].clone()
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                }
            selected = self.participants(round_id)
            updates = [
                self.client_update(
                    cid,
                    state,
                    round_id,
                    caches[cid],
                    request_reference=self.request_reference,
                )
                for cid in selected
            ]
            states, counts, local_metrics = map(list, zip(*updates))
            if args.method == "efu" and args.aggregation == "encrypted":
                from .utils.fe import fe_secure_aggregate

                updated = fe_secure_aggregate(
                    states,
                    counts,
                    kappa=args.kappa,
                    precision=args.precision,
                    max_abs=args.max_abs,
                    seed=args.seed,
                    round_id=round_id,
                )
                if args.unlearn_method == "pgd" and round_id == args.unlearn_start - 1:
                    from .utils.fe import quantized_contribution

                    for cid, local, count in zip(selected, states, counts):
                        transmitted = quantized_contribution(
                            local,
                            kappa=args.kappa,
                            precision=args.precision,
                            seed=args.seed,
                        )
                        observe_contribution(
                            caches[cid], transmitted, round_id, count / sum(counts)
                        )
            else:
                updated = self.aggregate_plaintext(
                    states, counts, selected, state, round_id, metrics=local_metrics
                )
            drift = sum(
                float((updated[n].float() - state[n].float()).square().sum())
                for n, p in self.model.named_parameters()
            ) / sum(p.numel() for p in self.model.parameters())
            state = updated
            row = {
                "round": round_id,
                "phase": self.phase(round_id),
                "participants": selected,
                "sample_counts": counts,
                "unlearning_clients": [
                    cid
                    for cid in selected
                    if self.is_unlearning(round_id) and cid in self.forget_by_client
                ],
                "parameter_mse": drift,
                "local_objectives": {
                    cid: {k: v for k, v in values.items() if not k.startswith("_")}
                    for cid, values in zip(selected, local_metrics)
                },
                **self.evaluate(state),
            }
            history.append(row)
            print(json.dumps(row))
        result = self.result(history)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
            checkpoint = {
                "model": state,
                "configuration": vars(args),
                "classes": self.classes,
                "initial_model_sha256": self.initial_model_sha256,
            }
            if self.task:
                checkpoint.update(
                    task_annotations=self.annotation_metadata,
                    task_recipe=result["task_recipe"],
                )
            torch.save(checkpoint, path.with_suffix(".pt"))
        return result
