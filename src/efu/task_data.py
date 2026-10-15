"""Explicit, user-supplied annotation contract for the LFW two-task experiment."""

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .dataset_io import digest


def prepare_gender_annotations(source, stage, annotations):
    annotations = Path(annotations)
    labels = {}
    with annotations.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "gender"]:
            raise ValueError("Gender CSV header must be path,gender")
        for row in reader:
            if set(row) != {"path", "gender"} or any(
                not isinstance(row[key], str) for key in ("path", "gender")
            ):
                raise ValueError("Every CSV row must contain exactly path,gender")
            relative = Path(row["path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 2
            ):
                raise ValueError(
                    "Annotation path must be identity/original-image-filename"
                )
            key = relative.as_posix()
            value = row["gender"].strip()
            if not value:
                raise ValueError(f"Empty supplied gender annotation: {key}")
            if key in labels:
                raise ValueError(f"Duplicate annotation: {key}")
            labels[key] = value
    selected = {}
    for split in ("train", "test"):
        for filename, _ in ImageFolder(stage / split).samples:
            relative = Path(filename).relative_to(stage)
            original = Path(*relative.parts[1:]).as_posix()
            if original not in labels:
                raise ValueError(f"Missing supplied gender annotation: {original}")
            selected[relative.as_posix()] = labels[original]
    categories = sorted(set(selected.values()))
    if len(categories) != 2:
        raise ValueError(
            "The paper gender task requires exactly two supplied categories across selected images"
        )
    mapping = {name: i for i, name in enumerate(categories)}
    metadata = {
        "annotation_source": str(annotations.resolve()),
        "annotation_sha256": digest(annotations),
        "gender_to_idx": mapping,
        "annotation_rows": len(labels),
        "selected_images": len(selected),
        "label_origin": "explicit user-supplied per-image CSV; no gender inference",
    }
    output = {
        "schema_version": 1,
        "metadata": metadata,
        "labels": {path: mapping[value] for path, value in selected.items()},
    }
    (stage / "tasks.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def load_task_annotations(directory, manifest):
    directory = Path(directory)
    path = directory / "tasks.json"
    expected = manifest.get("files", {}).get("tasks.json")
    if not path.is_file() or not expected or "tasks" not in manifest:
        raise ValueError(
            "Missing tasks.json annotations; prepare LFW with --gender-annotations"
        )
    if digest(path) != expected:
        raise ValueError("Task annotation hash changed from the prepared manifest")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("metadata") != manifest["tasks"]:
        raise ValueError("Task annotation schema/provenance differs from manifest")
    mapping = data["metadata"]["gender_to_idx"]
    if sorted(mapping.values()) != [0, 1]:
        raise ValueError("Expected two supplied gender categories indexed 0 and 1")
    expected_paths = {
        Path(filename).relative_to(directory).as_posix()
        for split in ("train", "test")
        for filename, _ in ImageFolder(directory / split).samples
    }
    if set(data["labels"]) != expected_paths:
        raise ValueError(
            "Task annotation coverage differs from prepared train/test images"
        )
    if any(
        type(value) is not int or value not in (0, 1)
        for value in data["labels"].values()
    ):
        raise ValueError("Invalid supplied gender class index")
    return data


class IdentityGenderDataset(Dataset):
    def __init__(self, base, gender):
        if len(base) != len(gender):
            raise ValueError("Gender annotation count does not match images")
        self.base = base
        self.gender = gender
        self.targets = getattr(base, "targets", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, identity = self.base[index]
        return image, torch.tensor(
            [int(identity), int(self.gender[index])], dtype=torch.long
        )


def attach_task_labels(train, test, root):
    directory = Path(root) / "lfw"
    manifest = json.loads((directory / "manifest.json").read_text())
    annotations = load_task_annotations(directory, manifest)
    wrapped = []
    for base in (train, test):
        gender = [
            annotations["labels"][Path(filename).relative_to(directory).as_posix()]
            for filename, _ in base.samples
        ]
        wrapped.append(IdentityGenderDataset(base, gender))
    return *wrapped, annotations["metadata"]
