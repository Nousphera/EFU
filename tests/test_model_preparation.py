import importlib.util
import pytest


def test_pretrained_preparation_api_exists():
    assert importlib.util.find_spec("efu.model_io") is not None, (
        "EFU needs an offline prepared-model loader"
    )


def test_missing_pretrained_model_never_falls_back_to_random(tmp_path):
    from efu.model_io import load_backbone

    with pytest.raises(FileNotFoundError, match="prepare_model"):
        load_backbone("resnet18", 10, root=tmp_path)


def test_explicit_scratch_model_keeps_full_backbone(tmp_path):
    from efu.model_io import load_backbone

    model = load_backbone("resnet18", 62, root=tmp_path, pretrained=False)
    assert model.fc.out_features == 62
    assert [len(getattr(model, f"layer{i}")) for i in range(1, 5)] == [2, 2, 2, 2]


def test_model_validation_rejects_modified_checkpoint(tmp_path):
    import json
    from efu.model_io import validate_model

    folder = tmp_path / "resnet18"
    folder.mkdir()
    (folder / "weights.pth").write_bytes(b"altered checkpoint")
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "resnet18",
                "weights": "ResNet18_Weights.IMAGENET1K_V1",
                "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            }
        )
    )
    with pytest.raises(ValueError, match="differ"):
        validate_model("resnet18", tmp_path)


def test_failed_model_download_never_publishes_prepared_root(tmp_path, monkeypatch):
    import torch
    from efu.model_io import prepare_model

    def broken_download(url, path, **options):
        path.write_bytes(b"partial")
        raise OSError("connection failed")

    monkeypatch.setattr(torch.hub, "download_url_to_file", broken_download)
    with pytest.raises(OSError, match="connection failed"):
        prepare_model("resnet18", tmp_path)
    assert not (tmp_path / "resnet18").exists()
    assert list(tmp_path.iterdir()) == []
