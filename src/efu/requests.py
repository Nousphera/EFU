"""Seeded request manifests, independent of minibatch RNG state."""

import math
import numpy as np


def build_requests(
    partitions,
    targets,
    clients,
    fraction,
    *,
    seed,
    forget_classes=None,
    class_fraction=None,
):
    if (
        not clients
        or len(set(clients)) != len(clients)
        or any(i < 0 or i >= len(partitions) for i in clients)
    ):
        raise ValueError("Requesters must be distinct valid client IDs")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Forget fraction must lie in (0,1]")
    labels = np.asarray(targets)
    universe = sorted(np.unique(labels).tolist())
    rng = np.random.default_rng(seed)
    classes = sorted(set(forget_classes or []))
    if class_fraction is not None:
        if classes or not 0 < class_fraction <= 1:
            raise ValueError("Choose explicit classes or a class fraction in (0,1]")
        classes = sorted(
            rng.choice(
                universe, max(1, int(len(universe) * class_fraction)), replace=False
            ).tolist()
        )
    if any(c not in universe for c in classes):
        raise ValueError("A forgotten class is absent from the dataset")
    forget, retain = {}, [list(indices) for indices in partitions]
    for cid in clients:
        original = partitions[cid]
        indices = (
            [i for i in original if int(labels[i]) in classes]
            if classes
            else sorted(
                rng.choice(
                    original, max(1, int(len(original) * fraction)), replace=False
                ).tolist()
            )
        )
        if not indices:
            raise ValueError(
                f"Client {cid} has no samples of the selected forgotten classes"
            )
        forget[cid] = indices
        removed = set(indices)
        retain[cid] = [i for i in original if i not in removed]
    return forget, retain, classes
