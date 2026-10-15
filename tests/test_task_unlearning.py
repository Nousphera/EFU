"""Identity-forgetting / supplied gender-retention route from paper section 5.2."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def lfw_fixture(root):
    source = root / "source"
    rows = []
    rng = np.random.default_rng(9)
    for identity in ("person_a", "person_b"):
        folder = source / identity
        folder.mkdir(parents=True)
        for i in range(8):
            name = f"{identity}/{i}.png"
            Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(
                source / name
            )
            rows.append((name, "annotation_a" if i % 2 else "annotation_b"))
    annotations = root / "supplied.csv"
    with annotations.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "gender"])
        writer.writerows(rows)
    return source, annotations


def test_annotation_import_is_explicit_complete_and_integrity_checked(tmp_path):
    from efu.dataset_io import prepare_dataset, validate_dataset, load_prepared
    from efu.task_unlearning import attach_task_labels
    from torchvision.transforms import ToTensor

    source, annotations = lfw_fixture(tmp_path)
    root = tmp_path / "data"
    prepare_dataset(
        "lfw", root, source=source, identities=2, gender_annotations=annotations
    )
    manifest = validate_dataset("lfw", root)
    assert manifest["tasks"]["gender_to_idx"] == {"annotation_a": 0, "annotation_b": 1}
    train, test, classes = load_prepared("lfw", root, ToTensor())
    train, test, metadata = attach_task_labels(train, test, root)
    assert train[0][1].shape == (2,)
    assert len(train) == 12 and len(test) == 4
    assert metadata["annotation_sha256"]
    path = root / "lfw/tasks.json"
    path.write_text(
        path.read_text().replace('"schema_version": 1', '"schema_version": 2')
    )
    with pytest.raises(ValueError, match="hash|changed"):
        attach_task_labels(train, test, root)


def test_annotation_missing_duplicate_and_invented_label_sources_fail(tmp_path):
    from efu.dataset_io import prepare_dataset

    source, annotations = lfw_fixture(tmp_path)
    rows = annotations.read_text().splitlines()
    annotations.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="Missing.*annotation"):
        prepare_dataset(
            "lfw",
            tmp_path / "missing",
            source=source,
            identities=2,
            gender_annotations=annotations,
        )
    assert not (tmp_path / "missing/lfw").exists()
    annotations.write_text("\n".join(rows + [rows[1]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_dataset(
            "lfw",
            tmp_path / "duplicate",
            source=source,
            identities=2,
            gender_annotations=annotations,
        )
    with pytest.raises(ValueError, match="LFW|lfw"):
        prepare_dataset(
            "imagefolder",
            tmp_path / "other",
            source=source,
            gender_annotations=annotations,
        )


def test_identity_forgetting_keeps_gender_gradient_and_reuses_offline_attack():
    from efu.task_unlearning import (
        DualHeadModel,
        prepare_task_unlearning,
        task_objective,
        unlearn_task,
    )

    torch.manual_seed(8)
    model = DualHeadModel("tiny", 3, image=False)
    data = DataLoader(
        TensorDataset(
            torch.rand(6, 4),
            torch.tensor([[0, 0], [1, 1], [2, 0], [0, 1], [1, 0], [2, 1]]),
        ),
        batch_size=3,
    )
    prepared = prepare_task_unlearning(model, data, attack_steps=2)
    batch = prepared["batches"][0]
    parts = task_objective(model, batch, prepared)
    g_expected = torch.autograd.grad(
        parts["retain"], model.gender_head.weight, retain_graph=True
    )[0]
    g_actual = torch.autograd.grad(sum(parts.values()), model.gender_head.weight)[0]
    assert g_actual.abs().sum() > 0
    torch.testing.assert_close(g_actual, g_expected)
    assert set(prepared["importance"]).isdisjoint(
        {"gender_head.weight", "gender_head.bias"}
    )
    attacks = [b["adversarial"].clone() for b in prepared["batches"]]
    before = model.gender_head.weight.detach().clone()
    unlearn_task(model, prepared, epochs=2, lr=0.01)
    assert not torch.equal(before, model.gender_head.weight)
    assert all(
        torch.equal(old, batch["adversarial"])
        for old, batch in zip(attacks, prepared["batches"])
    )
    assert all(
        torch.all(b["identity"] != b["adversarial_identity"])
        for b in prepared["batches"]
    )


def test_retained_only_training_never_updates_identity_head():
    from efu.task_unlearning import DualHeadModel, fit_tasks

    model = DualHeadModel("tiny", 3, image=False)
    data = DataLoader(
        TensorDataset(torch.rand(4, 4), torch.tensor([[0, 0], [1, 1], [2, 0], [0, 1]])),
        batch_size=4,
    )
    identity = model.identity_head.weight.detach().clone()
    gender = model.gender_head.weight.detach().clone()
    fit_tasks(model, data, epochs=2, lr=0.01, retain_only=True)
    torch.testing.assert_close(identity, model.identity_head.weight, rtol=0, atol=0)
    assert not torch.equal(gender, model.gender_head.weight)


def test_primary_task_path_runs_all_phases_with_real_encryption(tmp_path):
    from efu.dataset_io import prepare_dataset
    from efu.train import main

    source, annotations = lfw_fixture(tmp_path)
    prepare_dataset(
        "lfw",
        tmp_path / "data",
        source=source,
        identities=2,
        gender_annotations=annotations,
    )
    output = tmp_path / "task.json"
    result = main(
        [
            "--task",
            "identity-gender",
            "--dataset",
            "lfw",
            "--data-root",
            str(tmp_path / "data"),
            "--model",
            "tiny",
            "--clients",
            "2",
            "--alpha",
            "10",
            "--rounds",
            "3",
            "--unlearn-start",
            "2",
            "--unlearn-rounds",
            "1",
            "--attack-steps",
            "2",
            "--precision",
            "3",
            "--kappa",
            "16",
            "--output",
            str(output),
        ]
    )
    assert [row["phase"] for row in result["history"]] == [
        "joint-learning",
        "identity-unlearning",
        "gender-only",
    ]
    assert result["history"][1]["unlearning_clients"] == [0, 1]
    assert all(
        0 <= row["test"]["identity"]["accuracy"] <= 1
        and 0 <= row["test"]["gender"]["accuracy"] <= 1
        for row in result["history"]
    )
    assert len({row["test"]["gender"]["loss"] for row in result["history"]}) == 3
    assert result["protocol"] == "decentralized-coordinate-fe"
    assert result["task_recipe"]["loss_weights"] == {
        "retain_gender": 1.0,
        "forget_identity": 1.0,
        "adversarial_identity": 1.0,
        "inverted_mas": 1.0,
    }
    checkpoint = torch.load(
        output.with_suffix(".pt"), weights_only=True, map_location="cpu"
    )
    assert (
        "identity_head.weight" in checkpoint["model"]
        and "gender_head.weight" in checkpoint["model"]
    )
    assert result["benchmark_reproduction"] is False


def test_task_path_rejects_partial_cohort_and_missing_annotations(tmp_path):
    from efu.train import main

    with pytest.raises(SystemExit):
        main(["--task", "identity-gender", "--clients", "4", "--participation", ".5"])
    source, _ = lfw_fixture(tmp_path)
    from efu.dataset_io import prepare_dataset

    prepare_dataset("lfw", tmp_path / "data", source=source, identities=2)
    with pytest.raises(ValueError, match="annotation|tasks.json"):
        main(
            [
                "--task",
                "identity-gender",
                "--dataset",
                "lfw",
                "--data-root",
                str(tmp_path / "data"),
                "--alpha",
                "10",
            ]
        )


@pytest.mark.parametrize("name,size", [("resnet18", 32), ("vit_b_32", 224)])
def test_supported_image_backbones_share_features_with_two_real_heads(name, size):
    from efu.task_unlearning import DualHeadModel

    torch.set_num_threads(1)
    model = DualHeadModel(name, identities=3, pretrained=False).eval()
    with torch.no_grad():
        result = model(torch.rand(1, 3, size, size))
    assert result["identity"].shape == (1, 3)
    assert result["gender"].shape == (1, 2)
    assert all(torch.isfinite(value).all() for value in result.values())


def test_gender_only_retraining_control_uses_no_identity_supervision():
    from efu.train import main

    result = main(["--task", "identity-gender", "--method", "retrain", "--rounds", "2"])
    assert all(
        row["phase"] == "gender-only" and not row["unlearning_clients"]
        for row in result["history"]
    )
    assert result["protocol"] == "plaintext-baseline"
