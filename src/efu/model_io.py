"""Pinned torchvision ImageNet checkpoints prepared once and loaded offline.

These explicit official weights are the release initialization. The paper names
Hugging Face ImageNet weights without identifying its checkpoint; equivalence to
that unpublished identifier is not asserted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

import torch

MODELS = ("resnet18", "vit_b_32")
WEIGHTS = {
    "resnet18": (
        "ResNet18_Weights.IMAGENET1K_V1",
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    ),
    "vit_b_32": (
        "ViT_B_32_Weights.IMAGENET1K_V1",
        "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    ),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _architecture(name):
    from torchvision import models

    if name not in MODELS:
        raise ValueError(f"Unsupported model: {name}")
    return getattr(models, name)(weights=None)


def validate_model(name, root="models"):
    if name not in MODELS:
        raise ValueError(f"Unsupported model: {name}")
    directory = Path(root) / name
    metadata = directory / "manifest.json"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"Missing {metadata}; run python scripts/prepare_model.py --model {name} --root {root}"
        )
    manifest = json.loads(metadata.read_text())
    identifier, url = WEIGHTS[name]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model") != name
        or manifest.get("weights") != identifier
        or manifest.get("url") != url
    ):
        raise ValueError("Prepared model identity differs from the pinned registry")
    checkpoint = directory / "weights.pth"
    if directory.is_symlink() or checkpoint.is_symlink() or metadata.is_symlink():
        raise ValueError("Prepared model files must not be symlinks")
    if not checkpoint.is_file() or digest(checkpoint) != manifest.get("sha256"):
        raise ValueError("Prepared model weights differ from their manifest")
    expected_prefix = re.search(r"-([0-9a-f]+)\.pth$", url).group(1)
    if not manifest["sha256"].startswith(expected_prefix):
        raise ValueError(
            "Prepared model digest differs from the official checkpoint hash"
        )
    return manifest


def prepare_model(name, root="models"):
    if name not in MODELS:
        raise ValueError(f"Unsupported model: {name}")
    parent = Path(root)
    destination = parent / name
    if destination.exists():
        return validate_model(name, parent)
    parent.mkdir(parents=True, exist_ok=True)
    identifier, url = WEIGHTS[name]
    expected_prefix = re.search(r"-([0-9a-f]+)\.pth$", url).group(1)
    with tempfile.TemporaryDirectory(prefix=f".{name}-prepare-", dir=parent) as work:
        stage = Path(work) / name
        stage.mkdir()
        checkpoint = stage / "weights.pth"
        torch.hub.download_url_to_file(
            url, checkpoint, hash_prefix=expected_prefix, progress=True
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Nonfinite tensor in pretrained checkpoint")
        model = _architecture(name)
        model.load_state_dict(state, strict=True)
        manifest = {
            "schema_version": 1,
            "model": name,
            "weights": identifier,
            "url": url,
            "sha256": digest(checkpoint),
            "corpus": "ImageNet-1K",
            "provider": "torchvision official explicit IMAGENET1K_V1",
            "paper_checkpoint_identity": "not specified in paper; release initialization",
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    return manifest


def load_backbone(name, classes, *, root="models", pretrained=True):
    if type(classes) is not int or classes < 2:
        raise ValueError("Classification requires at least two classes")
    manifest = validate_model(name, root) if pretrained else None
    model = _architecture(name)
    if manifest is not None:
        state = torch.load(
            Path(root) / name / "weights.pth", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
    if name == "resnet18":
        model.fc = torch.nn.Linear(model.fc.in_features, classes)
    else:
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, classes)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=MODELS, help="Omit to prepare both paper backbones"
    )
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    reports = []
    for name in (args.model,) if args.model else MODELS:
        try:
            result = (validate_model if args.validate else prepare_model)(
                name, args.root
            )
            reports.append(
                {
                    "model": name,
                    "root": str(args.root / name),
                    "sha256": result["sha256"],
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(2, f"Model preparation failed: {exc}\n")
    print(json.dumps(reports, indent=2))
    return reports
