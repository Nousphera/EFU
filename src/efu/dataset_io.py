"""Prepare immutable dataset roots shared by the CLI and actual training adapters.

Public downloads use torchvision's checksum-verified dataset implementations.
Imports are extracted directories; this module never unpacks user archives.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
from torchvision import datasets as D
from torchvision.datasets.folder import IMG_EXTENSIONS

DATASETS = ("cifar10", "cifar100", "lfw", "femnist", "imagefolder")
FOLDERS = ("imagefolder", "lfw", "eurosat", "cars196")
SOURCES = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "femnist": "https://github.com/TalwalkarLab/leaf/tree/master/data/femnist",
    "lfw": "https://vis-www.cs.umass.edu/lfw/",
    "svhn": "http://ufldl.stanford.edu/housenumbers/",
    "fashion-mnist": "https://github.com/zalandoresearch/fashion-mnist",
    "emnist": "https://www.nist.gov/itl/products-and-services/emnist-dataset",
    "food101": "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Dataset roots must not be symlinks")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in datasets: {path}")
        if path.is_file() and path != root / "manifest.json":
            files[path.relative_to(root).as_posix()] = digest(path)
    if not files:
        raise ValueError("Dataset is empty")
    return files


def labels(dataset):
    for name in ("targets", "labels", "_labels"):
        if hasattr(dataset, name):
            return np.asarray(getattr(dataset, name), dtype=np.int64)
    return np.asarray([label for _, label in dataset.samples], dtype=np.int64)


def open_dataset(name, root, transform=None, download=False):
    """root is the dataset directory, e.g. data/cifar10, not its parent."""
    root = Path(root)
    if name in ("cifar10", "cifar100", "fashion-mnist", "emnist"):
        cls = {
            "cifar10": D.CIFAR10,
            "cifar100": D.CIFAR100,
            "fashion-mnist": D.FashionMNIST,
            "emnist": D.EMNIST,
        }[name]
        extra = {"split": "balanced"} if name == "emnist" else {}
        train = cls(root, train=True, transform=transform, download=download, **extra)
        test = cls(root, train=False, transform=transform, download=download, **extra)
    elif name in ("svhn", "food101"):
        cls = D.SVHN if name == "svhn" else D.Food101
        train = cls(root, split="train", transform=transform, download=download)
        test = cls(root, split="test", transform=transform, download=download)
    elif name in FOLDERS:
        if download:
            raise ValueError(f"{name} requires user-acquired --source data")
        train = D.ImageFolder(root / "train", transform=transform)
        test = D.ImageFolder(root / "test", transform=transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("Train/test class directories differ")
    elif name == "femnist":
        from .femnist import FEMNIST

        train, test = [FEMNIST(root / split, transform) for split in ("train", "test")]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test


def describe(name, root, *, verify_images=False):
    train, test = open_dataset(name, root)
    class_to_idx = getattr(train, "class_to_idx", None)
    if class_to_idx is None:
        classes = getattr(train, "classes", [str(i) for i in range(10)])
        class_to_idx = {str(label): i for i, label in enumerate(classes)}
    splits = {}
    image_hashes = []
    for split, dataset in (("train", train), ("test", test)):
        y = labels(dataset)
        if (
            len(y) == 0
            or y.shape != (len(dataset),)
            or y.min() < 0
            or y.max() >= len(class_to_idx)
        ):
            raise ValueError(f"Invalid {split} label range or sample count")
        splits[split] = {
            "samples": len(y),
            "labels_sha256": hashlib.sha256(y.astype("<i8").tobytes()).hexdigest(),
        }
        if hasattr(dataset, "users"):
            splits[split]["users"] = dataset.users
        if name in FOLDERS and verify_images:
            hashes = set()
            for filename, _ in dataset.samples:
                with Image.open(filename) as image:
                    image.verify()
                hashes.add(digest(filename))
            image_hashes.append(hashes)
    if len(image_hashes) == 2 and image_hashes[0] & image_hashes[1]:
        raise ValueError("Exact duplicate image content appears in train and test")
    return {"class_to_idx": class_to_idx, "splits": splits}


def validate_dataset(name, root):
    directory = Path(root) / name
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != name:
        raise ValueError("Manifest schema/dataset mismatch")
    if inventory(directory) != manifest["files"]:
        raise ValueError(
            "Dataset files differ from manifest; do not silently replace prepared data"
        )
    actual = describe(name, directory, verify_images=True)
    if any(actual[key] != manifest[key] for key in actual):
        raise ValueError("Dataset labels or split sizes differ from manifest")
    return manifest


def load_prepared(name, root, transform=None):
    directory = Path(root) / name
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run scripts/prepare_data.py first")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != name:
        raise ValueError("Manifest schema/dataset mismatch")
    train, test = open_dataset(name, directory, transform=transform)
    for split, dataset in (("train", train), ("test", test)):
        y = labels(dataset)
        if (
            len(dataset) != manifest["splits"][split]["samples"]
            or hashlib.sha256(y.astype("<i8").tobytes()).hexdigest()
            != manifest["splits"][split]["labels_sha256"]
        ):
            raise ValueError("Dataset labels or split sizes differ from manifest")
    mapping = getattr(train, "class_to_idx", manifest["class_to_idx"])
    if mapping != manifest["class_to_idx"]:
        raise ValueError("Dataset class mapping differs from manifest")
    return train, test, len(manifest["class_to_idx"])


def manifest_digest(name, root):
    return None if name == "synthetic" else digest(Path(root) / name / "manifest.json")


def copy_source(name, source, stage):
    """Copy only the files consumed by this dataset, never unrelated source data."""
    entries = {
        "cifar10": ["cifar-10-batches-py"],
        "cifar100": ["cifar-100-python"],
        "svhn": ["train_32x32.mat", "test_32x32.mat"],
        "fashion-mnist": ["FashionMNIST"],
        "emnist": ["EMNIST"],
        "food101": ["food-101"],
    }.get(name, ["train", "test"])
    stage.mkdir()
    for entry in entries:
        path = source / entry
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in datasets: {path}")
        if path.is_dir():
            inventory(path)
            shutil.copytree(path, stage / entry)
        elif path.is_file():
            shutil.copy2(path, stage / entry)
        else:
            raise FileNotFoundError(f"Missing dataset source: {path}")


def prepare_dataset(
    name,
    root,
    *,
    source=None,
    download=False,
    identities=30,
    split_seed=2025,
    test_fraction=0.2,
):
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    parent = Path(root)
    destination = parent / name
    if destination.exists():
        raise FileExistsError(
            f"{destination} exists; use --validate or choose another root"
        )
    if bool(source) == bool(download):
        raise ValueError("Choose exactly one of --source or --download")
    if source and (
        Path(source).resolve() == parent.resolve()
        or Path(source).resolve() in parent.resolve().parents
    ):
        raise ValueError("Output root must not be inside the imported source")
    parent.mkdir(parents=True, exist_ok=True)
    preparation = {"mode": "download" if download else "import"}
    with tempfile.TemporaryDirectory(prefix=f".{name}-prepare-", dir=parent) as work:
        stage = Path(work) / name
        if source:
            source = Path(source)
            if source.is_symlink() or not source.is_dir():
                raise ValueError("Source must be a real extracted directory")
            if name == "lfw":
                inventory(source)
                prepare_lfw(source, stage, identities, split_seed, test_fraction)
                preparation.update(
                    identities=identities,
                    split_seed=split_seed,
                    test_fraction=test_fraction,
                    selection="most-images-first; ties by identity name",
                )
            else:
                copy_source(name, source, stage)
                (stage / "manifest.json").unlink(missing_ok=True)
        else:
            if name in FOLDERS or name == "femnist":
                raise ValueError(
                    f"{name} requires --source; see dataset acquisition instructions"
                )
            stage.mkdir()
            open_dataset(name, stage, download=True)
        manifest = {
            "schema_version": 1,
            "dataset": name,
            "source": SOURCES.get(name, "user-supplied fixed train/test image folders"),
            "preparation": preparation,
            **describe(name, stage, verify_images=True),
            "files": inventory(stage),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    return manifest


def prepare_lfw(source, stage, identities, seed, fraction):
    if identities < 2 or not 0 < fraction < 1 or not 0 <= seed < 2**32:
        raise ValueError("LFW needs >=2 identities, uint32 seed and 0<test-fraction<1")
    candidates = []
    for folder in source.iterdir():
        if folder.is_dir():
            images = sorted(
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
            )
            if len(images) >= 2:
                candidates.append((folder.name, images))
    candidates.sort(key=lambda entry: (-len(entry[1]), entry[0]))
    if len(candidates) < identities:
        raise ValueError(
            f"Need {identities} identities with at least two images; found {len(candidates)}"
        )
    rng = np.random.default_rng(seed)
    for identity, images in candidates[:identities]:
        order = rng.permutation(len(images))
        ntest = min(len(images) - 1, max(1, round(len(images) * fraction)))
        for split, indices in (("test", order[:ntest]), ("train", order[ntest:])):
            folder = stage / split / identity
            folder.mkdir(parents=True)
            for index in indices:
                shutil.copy2(images[index], folder / images[index].name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent root used by the trainer; output is ROOT/DATASET",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source",
        type=Path,
        help="Extracted source root; no archive extraction or overwrite",
    )
    mode.add_argument(
        "--download",
        action="store_true",
        help="Explicit public download using torchvision and its checksums",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Read-only SHA256, split, class and image validation",
    )
    if "lfw" in DATASETS:
        parser.add_argument(
            "--identities",
            type=int,
            default=30,
            help="LFW: most populous identities, alphabetic tie break",
        )
        parser.add_argument("--split-seed", type=int, default=2025)
        parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        if args.validate:
            result = validate_dataset(args.dataset, args.root)
        else:
            result = prepare_dataset(
                args.dataset,
                args.root,
                source=args.source,
                download=args.download,
                identities=getattr(args, "identities", 30),
                split_seed=getattr(args, "split_seed", 2025),
                test_fraction=getattr(args, "test_fraction", 0.2),
            )
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Dataset preparation failed: {exc}\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "root": str(args.root / args.dataset),
                "splits": result["splits"],
            },
            indent=2,
        )
    )
