"""Paper-grounded experiment configuration and local integration runner."""

import argparse
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT = "EFU"


class ImageModel(torch.nn.Module):
    def __init__(self, name, classes, pretrained=False, model_root="models"):
        super().__init__()
        from .model_io import load_backbone

        self.body = load_backbone(name, classes, root=model_root, pretrained=pretrained)
        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)
        self.label2class = {i: i for i in range(classes)}

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        mean = x.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = x.new_tensor(self.input_std).view(1, 3, 1, 1)
        return self.body((x - mean) / std)


def partition(targets, clients, alpha, seed):
    """Class-wise Dirichlet partitions; reject empty clients explicitly."""
    if clients < 2 or alpha <= 0:
        raise ValueError("Use at least two clients and positive Dirichlet alpha")
    rng = np.random.default_rng(seed)
    result = [[] for _ in range(clients)]
    labels = np.asarray(targets)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.repeat(alpha, clients))
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for subset, indices in zip(result, np.split(indices, cuts)):
            subset.extend(indices.tolist())
    if any((not x for x in result)):
        raise ValueError(
            "Empty Dirichlet client; increase data/alpha or choose another seed"
        )
    return result


def load_data(args):
    from torchvision import transforms
    from .dataset_io import load_prepared, prepare_dataset, labels

    size = 224
    transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    if args.download:
        prepare_dataset(args.dataset, args.data_root, download=True)
    train, test, classes = load_prepared(args.dataset, args.data_root, transform)
    targets = labels(train)
    if getattr(args, "partition", "dirichlet") == "writer":
        if args.dataset != "femnist":
            raise ValueError("Writer partitioning is only defined for FEMNIST")
        if args.clients != len(train.users):
            raise ValueError(
                f"Writer partitioning needs --clients {len(train.users)} (one per training writer)"
            )
        subsets = train.user_indices
    else:
        subsets = partition(targets, args.clients, args.alpha, args.partition_seed)
    return train, test, subsets, targets, classes


def loader(dataset, indices, args, shuffle=True):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def fit(model, data, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        for x, y in data:
            x, y = (x.to(model.device), y.to(model.device))
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, data):
    model.eval()
    correct, total, loss = (0, 0, 0.0)
    with torch.no_grad():
        for x, y in data:
            x, y = (x.to(model.device), y.to(model.device))
            logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
            loss += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
    return (
        {"accuracy": correct / total, "loss": loss / total, "samples": total}
        if total
        else {"accuracy": None, "loss": None, "samples": 0}
    )


def build_parser():
    p = argparse.ArgumentParser(
        description="Paper-grounded EFU learning and unlearning"
    )
    p.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "imagefolder", "femnist", "lfw"],
        default="cifar10",
    )
    p.add_argument(
        "--task",
        choices=["classification", "identity-gender"],
        default="classification",
    )
    p.add_argument("--model", choices=["resnet18", "vit_b_32"], default="resnet18")
    p.add_argument("--model-root", default="models")
    p.add_argument(
        "--from-scratch",
        dest="pretrained",
        action="store_false",
        help="Explicit random-initialization ablation",
    )
    p.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help=argparse.SUPPRESS
    )
    p.set_defaults(pretrained=True)
    p.add_argument("--partition", choices=["dirichlet", "writer"], default="dirichlet")
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--participation", type=float, default=1.0)
    p.add_argument("--kappa", type=int, default=64)
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--max-abs", type=float, default=32.0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--partition-seed",
        type=int,
        default=2025,
        help="Fixed data allocation independent of model/training seed",
    )
    p.add_argument("--request-seed", type=int, default=2026)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--aggregation", choices=["encrypted", "plaintext"], default="encrypted"
    )
    p.add_argument("--method", choices=["efu", "fedavg", "retrain"], default="efu")
    p.add_argument(
        "--unlearn-method", choices=["efu", "pgd", "sga_ewc", "fedosd"], default="efu"
    )
    p.add_argument(
        "--variant",
        choices=["efu", "baseline"],
        default="efu",
        help="EFU adds MAS to PGD/FedOSD; SGA-EWC uses Fisher only",
    )
    p.add_argument(
        "--unlearn-start",
        type=int,
        default=51,
        help="First unlearning round, one-based",
    )
    p.add_argument("--unlearn-rounds", type=int, default=10)
    p.add_argument("--unlearn-epochs", type=int, default=5)
    p.add_argument(
        "--forget-client", type=int, default=0, help="Single-requester shorthand"
    )
    p.add_argument("--forget-clients", type=int, nargs="+", default=None)
    p.add_argument("--forget-fraction", type=float, default=0.1)
    p.add_argument(
        "--forget-class", type=int, default=None, help="Single-class shorthand"
    )
    p.add_argument("--forget-classes", type=int, nargs="+", default=None)
    p.add_argument("--forget-class-fraction", type=float, default=None)
    p.add_argument(
        "--request-participation", choices=["random", "required"], default="random"
    )
    p.add_argument("--attack-steps", type=int, default=100)
    p.add_argument("--attack-eps", type=float, default=0.4)
    p.add_argument("--mas-weight", type=float, default=1.0)
    p.add_argument("--ewc-weight", type=float, default=1.0)
    p.add_argument(
        "--fisher-source",
        choices=["forget", "retain"],
        default="forget",
        help="Declared empirical-Fisher data source",
    )
    p.add_argument(
        "--pgd-radius",
        type=float,
        default=3.0,
        help="Explicit L2 trust radius; original random-model calibration is not inferred",
    )
    return p


def validate_args(args):
    if (
        args.clients < 2
        or min(
            args.rounds,
            args.batch_size,
            args.local_epochs,
            args.kappa,
            args.unlearn_start,
            args.unlearn_rounds,
            args.unlearn_epochs,
            args.attack_steps,
        )
        < 1
    ):
        raise ValueError("Positive sizes/rounds and at least two clients are required")
    if (
        not 0 < args.participation <= 1
        or math.floor(args.clients * args.participation) < 2
    ):
        raise ValueError("Participation must select at least two clients")
    for name in ("lr", "alpha", "pgd_radius", "attack_eps", "max_abs"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in (args.mas_weight, args.ewc_weight)):
        raise ValueError("Regularizer weights must be finite and nonnegative")
    if not 0 <= args.precision <= 6 or not 0 < args.forget_fraction <= 1:
        raise ValueError("Invalid precision or forget fraction")
    requesters = (
        args.forget_clients if args.forget_clients is not None else [args.forget_client]
    )
    if (
        not requesters
        or len(set(requesters)) != len(requesters)
        or any(not 0 <= cid < args.clients for cid in requesters)
    ):
        raise ValueError("Requesters must be unique valid client IDs")
    if args.forget_classes is not None and args.forget_class is not None:
        raise ValueError("Use forget-class or forget-classes, not both")
    if args.forget_class_fraction is not None and (
        not 0 < args.forget_class_fraction <= 1
        or args.forget_classes is not None
        or args.forget_class is not None
    ):
        raise ValueError(
            "Choose explicit forgotten classes or a class fraction in (0,1]"
        )
    if (
        args.method == "efu"
        and args.unlearn_method == "fedosd"
        and args.aggregation != "plaintext"
    ):
        raise ValueError(
            "FedOSD requires plaintext cohort gradients; FE-wrapped FedOSD is unsupported"
        )
    if args.method == "efu" and args.unlearn_method == "pgd":
        if args.unlearn_start < 2:
            raise ValueError("PGD needs a preparation aggregation before unlearning")
        if (
            len(requesters) >= math.floor(args.clients * args.participation)
            and args.task == "classification"
        ):
            raise ValueError(
                "PGD needs a preparation cohort containing requesters and a retained participant"
            )
    if args.task == "identity-gender":
        if args.dataset != "lfw" or args.participation != 1.0:
            raise ValueError(
                "identity-gender requires LFW annotations and full participation"
            )
        if (
            args.forget_class is not None
            or args.forget_classes is not None
            or args.forget_class_fraction is not None
        ):
            raise ValueError(
                "Task forgetting uses all identities; class selection is a different request"
            )
        if (
            args.method == "efu"
            and args.unlearn_method == "sga_ewc"
            and args.fisher_source != "forget"
        ):
            raise ValueError(
                "Task SGA-EWC estimates identity Fisher; select fisher-source=forget"
            )
    if args.output:
        destination = Path(args.output)
        if destination.suffix != ".json":
            raise ValueError("--output must end in .json")
        if destination.exists() or destination.with_suffix(".pt").exists():
            raise FileExistsError(f"Output already exists: {destination}")
    return args


def main(argv=None):
    parser = build_parser()
    try:
        args = validate_args(parser.parse_args(argv))
    except ValueError as error:
        parser.error(str(error))
    from .experiment import Experiment

    experiment = Experiment(args)
    return experiment.run()


def cli():
    main()


if __name__ == "__main__":
    cli()
