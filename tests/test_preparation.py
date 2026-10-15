"""Data preparation must preserve split identity and feed the actual trainer."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def fixture_images(root):
    rng = np.random.default_rng(19)
    for split, n in (("train", 8), ("test", 3)):
        for label in ("a", "b"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(n):
                pixels = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i}.png")
    return root


def prepare(root, *args):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_data.py"),
            "--dataset",
            "imagefolder",
            "--root",
            str(root),
            *map(str, args),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_import_validate_and_reject_changed_files(tmp_path):
    source = fixture_images(tmp_path / "source")
    (source / "unrelated-private.txt").write_text("do not import")
    result = prepare(tmp_path / "data", "--source", source)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "data/imagefolder/manifest.json").read_text())
    assert manifest["splits"]["train"]["samples"] == 16
    assert manifest["splits"]["test"]["samples"] == 6
    assert manifest["class_to_idx"] == {"a": 0, "b": 1}
    assert not (tmp_path / "data/imagefolder/unrelated-private.txt").exists()
    assert prepare(tmp_path / "data", "--validate").returncode == 0
    assert prepare(tmp_path / "data", "--source", source).returncode != 0
    (tmp_path / "data/imagefolder/train/a/0.png").write_bytes(b"changed")
    assert prepare(tmp_path / "data", "--validate").returncode != 0


def test_reject_split_overlap_class_mismatch_and_symlinks(tmp_path):
    import shutil

    source = fixture_images(tmp_path / "source")
    shutil.copyfile(source / "train/a/0.png", source / "test/a/0.png")
    assert prepare(tmp_path / "data", "--source", source).returncode != 0
    assert not (tmp_path / "data/imagefolder").exists()
    (source / "test/a/0.png").unlink()
    (source / "test/b").rename(source / "test/c")
    assert prepare(tmp_path / "data", "--source", source).returncode != 0
    (source / "test/c").rename(source / "test/b")
    (source / "train/a/link.png").symlink_to(source / "test/a/1.png")
    assert prepare(tmp_path / "data", "--source", source).returncode != 0


def test_prepared_images_train_with_full_paper_architecture(tmp_path):
    from efu.train import main

    source = fixture_images(tmp_path / "source")
    result = prepare(tmp_path / "data", "--source", source)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "result.json"
    report = main(
        [
            "--dataset",
            "imagefolder",
            "--data-root",
            str(tmp_path / "data"),
            "--model",
            "resnet18",
            "--method",
            "fedavg",
            "--alpha",
            "10",
            "--rounds",
            "2",
            "--precision",
            "1",
            "--output",
            str(output),
        ]
    )
    assert len(report["history"]) == 2
    assert report["protocol"] == "plaintext-baseline"
    assert len(report["data_manifest_sha256"]) == 64
    assert sorted(i for part in report["partitions"] for i in part) == list(range(16))
    assert output.with_suffix(".pt").is_file()


def test_femnist_writer_json_feeds_actual_image_trainer(tmp_path):
    from efu.train import main

    source = tmp_path / "leaf"
    rng = np.random.default_rng(5)
    for split in ("train", "test"):
        folder = source / split
        folder.mkdir(parents=True)
        data = {
            user: {"x": rng.random((4, 784)).tolist(), "y": [0, 1, 0, 1]}
            for user in ("writer-a", "writer-b")
        }
        (folder / "shard.json").write_text(
            json.dumps({"users": list(data), "num_samples": [4, 4], "user_data": data})
        )
    cmd = [
        sys.executable,
        str(ROOT / "scripts/prepare_data.py"),
        "--dataset",
        "femnist",
        "--root",
        str(tmp_path / "data"),
        "--source",
        str(source),
    ]
    prep = subprocess.run(cmd, capture_output=True, text=True)
    assert prep.returncode == 0, prep.stderr
    result = main(
        [
            "--dataset",
            "femnist",
            "--data-root",
            str(tmp_path / "data"),
            "--partition",
            "writer",
            "--clients",
            "2",
            "--rounds",
            "1",
            "--method",
            "fedavg",
        ]
    )
    assert result["partitions"] == [list(range(4)), list(range(4, 8))]
    shard = json.loads((source / "train/shard.json").read_text())
    shard["user_data"]["writer-a"]["y"][0] = 62
    (source / "train/shard.json").write_text(json.dumps(shard))
    cmd[cmd.index("--root") + 1] = str(tmp_path / "bad")
    assert subprocess.run(cmd, capture_output=True).returncode != 0


def test_lfw_import_selects_declared_identities_and_preserves_split(tmp_path):
    source = fixture_images(tmp_path / "faces") / "train"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/prepare_data.py"),
        "--dataset",
        "lfw",
        "--root",
        str(tmp_path / "data"),
        "--source",
        str(source),
        "--identities",
        "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "data/lfw/manifest.json").read_text())
    assert manifest["splits"]["train"]["samples"] == 12
    assert manifest["splits"]["test"]["samples"] == 4
    assert manifest["preparation"]["split_seed"] == 2025
    from efu.train import main

    report = main(
        [
            "--dataset",
            "lfw",
            "--data-root",
            str(tmp_path / "data"),
            "--alpha",
            "10",
            "--rounds",
            "1",
            "--method",
            "fedavg",
        ]
    )
    assert sum(map(len, report["partitions"])) == 12


def test_preparation_rejects_image_with_valid_header_but_truncated_pixels(tmp_path):
    source = fixture_images(tmp_path / "source")
    image = source / "train/a/truncated.jpg"
    Image.new("RGB", (40, 40), (20, 40, 80)).save(image)
    image.write_bytes(image.read_bytes()[:-20])
    with Image.open(image) as header:
        header.verify()  # JPEG header verification alone accepts this file.
    result = prepare(tmp_path / "data", "--source", source)
    assert result.returncode != 0
    assert not (tmp_path / "data/imagefolder").exists()
