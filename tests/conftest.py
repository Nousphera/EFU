"""Bound CPU work without changing any tested architecture."""

import torch
import pytest


def pytest_sessionstart(session):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@pytest.fixture
def imagefolder_args(tmp_path):
    """Prepared image files exercise the same public loaders as real releases."""
    import numpy as np
    from PIL import Image
    from efu.dataset_io import prepare_dataset

    source = tmp_path / "images"
    rng = np.random.default_rng(19)
    for split, count in [("train", 8), ("test", 3)]:
        for label in range(3):
            folder = source / split / str(label)
            folder.mkdir(parents=True)
            for i in range(count):
                Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(
                    folder / f"{i}.png"
                )
    root = tmp_path / "data"
    prepare_dataset("imagefolder", root, source=source)
    return [
        "--dataset",
        "imagefolder",
        "--data-root",
        str(root),
        "--from-scratch",
        "--clients",
        "2",
        "--alpha",
        "10",
        "--batch-size",
        "8",
        "--rounds",
        "2",
        "--unlearn-start",
        "2",
        "--unlearn-rounds",
        "1",
        "--unlearn-epochs",
        "1",
        "--attack-steps",
        "1",
        "--aggregation",
        "plaintext",
    ]
