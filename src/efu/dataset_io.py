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

PAPER_DATASETS = ("cifar10", "lfw", "femnist")
DATASETS = ("cifar10", "cifar100", "lfw", "femnist", "imagefolder")
FOLDERS = ("imagefolder", "lfw")
SOURCES = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "femnist": "https://github.com/TalwalkarLab/leaf/tree/master/data/femnist",
    "lfw": "https://vis-www.cs.umass.edu/lfw/",
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


def open_dataset(name, root, transform=None, download=False, *, test_transform=None):
    """root is the dataset directory, e.g. data/cifar10, not its parent."""
    root = Path(root)
    test_transform = transform if test_transform is None else test_transform
    if name in ("cifar10", "cifar100"):
        cls = {
            "cifar10": D.CIFAR10,
            "cifar100": D.CIFAR100,
        }[name]
        train = cls(root, train=True, transform=transform, download=download)
        test = cls(root, train=False, transform=test_transform, download=download)
    elif name in FOLDERS:
        if download:
            raise ValueError(f"{name} requires user-acquired --source data")
        train = D.ImageFolder(root / "train", transform=transform)
        test = D.ImageFolder(root / "test", transform=test_transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("Train/test class directories differ")
    elif name == "femnist":
        from .femnist import FEMNIST

        train = FEMNIST(root / "train", transform)
        test = FEMNIST(root / "test", test_transform)
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
                    image.load()
                hashes.add(digest(filename))
            image_hashes.append(hashes)
    if (
        name == "femnist"
        and hasattr(train, "sample_ids")
        and set(train.sample_ids) & set(test.sample_ids)
    ):
        raise ValueError("FEMNIST source sample appears in both train and test")
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
    if "tasks" in manifest:
        from .task_data import load_task_annotations

        load_task_annotations(directory, manifest)
    return manifest


def load_prepared(name, root, transform=None, *, test_transform=None):
    directory = Path(root) / name
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run scripts/prepare_data.py first")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != name:
        raise ValueError("Manifest schema/dataset mismatch")
    validate_dataset(name, root)
    train, test = open_dataset(
        name, directory, transform=transform, test_transform=test_transform
    )
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
    return digest(Path(root) / name / "manifest.json")


def copy_source(name, source, stage):
    """Copy only the files consumed by this dataset, never unrelated source data."""
    entries = {
        "cifar10": ["cifar-10-batches-py"],
        "cifar100": ["cifar-100-python"],
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
    gender_annotations=None,
):
    if gender_annotations is not None and name != "lfw":
        raise ValueError("Gender annotations are only supported for LFW")
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    parent = Path(root)
    destination = parent / name
    if source and download:
        raise ValueError("Choose --source or --download, not both")
    if destination.exists():
        existing = validate_dataset(name, root)
        if name == "lfw" and any(
            existing["preparation"].get(k) != v
            for k, v in {
                "identities": identities,
                "split_seed": split_seed,
                "test_fraction": test_fraction,
            }.items()
        ):
            raise ValueError(
                "Existing LFW split differs from requested configuration; choose another root"
            )
        if name == "femnist" and existing["preparation"].get("mode") == "download":
            requested = {"split_seed": split_seed, "test_fraction": test_fraction}
            if any(existing["preparation"].get(k) != v for k, v in requested.items()):
                raise ValueError(
                    "Existing FEMNIST split differs from requested configuration; choose another root"
                )
        if gender_annotations is not None:
            expected = existing.get("tasks", {}).get("annotation_sha256")
            if expected != digest(gender_annotations):
                raise ValueError(
                    "Existing LFW annotations differ or are absent; choose a new output root"
                )
        return existing
    download = source is None
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
            if name == "lfw":
                from .acquisition import acquire_lfw

                source, provenance = acquire_lfw(parent / "raw")
                prepare_lfw(source, stage, identities, split_seed, test_fraction)
                preparation.update(
                    provenance,
                    identities=identities,
                    split_seed=split_seed,
                    test_fraction=test_fraction,
                    selection="most-images-first; ties by identity name",
                )
            elif name == "femnist":
                from .acquisition import acquire_femnist

                preparation.update(
                    acquire_femnist(
                        parent / "raw",
                        stage,
                        seed=split_seed,
                        test_fraction=test_fraction,
                    )
                )
            elif name in FOLDERS:
                raise ValueError(
                    f"{name} requires --source train/ and test/ image folders"
                )
            else:
                stage.mkdir()
                open_dataset(name, stage, download=True)
        task_metadata = None
        if gender_annotations is not None:
            from .task_data import prepare_gender_annotations

            task_metadata = prepare_gender_annotations(
                source, stage, gender_annotations
            )
        manifest = {
            "schema_version": 1,
            "dataset": name,
            "source": SOURCES.get(name, "user-supplied fixed train/test image folders"),
            "preparation": preparation,
            **describe(name, stage, verify_images=True),
            "files": inventory(stage),
        }
        if task_metadata is not None:
            manifest["tasks"] = task_metadata
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
    parser.add_argument(
        "--dataset", choices=DATASETS, help="Omit to prepare all paper datasets"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent root used by the trainer; output is ROOT/DATASET",
    )
    mode = parser.add_mutually_exclusive_group()
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
        parser.add_argument(
            "--gender-annotations",
            type=Path,
            help="LFW only: explicit CSV with path,gender for every selected image",
        )
        parser.add_argument("--split-seed", type=int, default=2025)
        parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.validate and getattr(args, "gender_annotations", None) is not None:
        parser.error(
            "--validate checks saved annotations; use a new output root to import annotations"
        )
    if args.source and args.dataset is None:
        parser.error("--source requires one --dataset")
    reports, failures = [], []
    for name in (args.dataset,) if args.dataset else PAPER_DATASETS:
        try:
            if args.validate:
                result = validate_dataset(name, args.root)
            else:
                options = dict(
                    source=args.source,
                    download=args.download,
                    identities=getattr(args, "identities", 30),
                    split_seed=getattr(args, "split_seed", 2025),
                    test_fraction=getattr(args, "test_fraction", 0.2),
                )
                options["gender_annotations"] = getattr(
                    args, "gender_annotations", None
                )
                result = prepare_dataset(name, args.root, **options)
            reports.append(
                {
                    "dataset": name,
                    "root": str(args.root / name),
                    "splits": result["splits"],
                }
            )
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            failures.append(f"{name}: {exc}")
    print(json.dumps(reports, indent=2))
    if failures:
        parser.exit(2, "Dataset preparation failed:\n" + "\n".join(failures) + "\n")
    return reports
