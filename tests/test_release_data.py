"""Release preparation contracts: real data, idempotence and immutable contents."""

import hashlib
import io
import tarfile
import pytest


@pytest.mark.parametrize(
    "name", ["fashion-mnist", "emnist", "svhn", "food101", "eurosat", "cars196"]
)
def test_unlisted_dataset_routes_are_rejected(name, tmp_path):
    from efu.dataset_io import open_dataset

    with pytest.raises(ValueError, match="Unknown dataset"):
        open_dataset(name, tmp_path)


def test_prepare_defaults_to_all_paper_datasets(tmp_path, monkeypatch, capsys):
    from efu import dataset_io

    prepared = []

    def acquire(name, root, **options):
        prepared.append(name)
        return {"splits": {"train": {"samples": 10}, "test": {"samples": 2}}}

    monkeypatch.setattr(dataset_io, "prepare_dataset", acquire)
    dataset_io.main(["--root", str(tmp_path)])
    assert prepared == ["cifar10", "lfw", "femnist"]
    prepared.clear()
    dataset_io.main(["--dataset", "cifar10", "--root", str(tmp_path)])
    assert prepared == ["cifar10"]


def test_repeat_preparation_validates_existing_content(tmp_path):
    from efu.dataset_io import prepare_dataset, load_prepared
    from test_preparation import fixture_images

    source = fixture_images(tmp_path / "source")
    root = tmp_path / "data"
    original = prepare_dataset("imagefolder", root, source=source)
    assert prepare_dataset("imagefolder", root, source=source) == original
    (root / "imagefolder/train/a/0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="files differ"):
        prepare_dataset("imagefolder", root, source=source)
    with pytest.raises(ValueError, match="files differ"):
        load_prepared("imagefolder", root)


def test_train_and_test_transform_are_independent(tmp_path):
    from efu.dataset_io import prepare_dataset, load_prepared
    from test_preparation import fixture_images

    prepare_dataset(
        "imagefolder", tmp_path / "data", source=fixture_images(tmp_path / "source")
    )
    train, test, classes = load_prepared(
        "imagefolder",
        tmp_path / "data",
        lambda x: "train",
        test_transform=lambda x: "test",
    )
    assert train[0][0] == "train" and test[0][0] == "test" and classes == 2


def test_lfw_archive_acquisition_preserves_original_images_and_reuses(
    tmp_path, monkeypatch
):
    import efu.acquisition as acquisition

    raw = tmp_path / "input.tgz"
    with tarfile.open(raw, "w:gz") as archive:
        payload = b"original-image-bytes"
        info = tarfile.TarInfo("lfw/person/photo.jpg")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    checksum = hashlib.sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setattr(acquisition, "LFW_SHA256", checksum)
    monkeypatch.setattr(acquisition, "LFW_URL", raw.as_uri())
    directory, provenance = acquisition.acquire_lfw(tmp_path / "raw")
    assert (directory / "person/photo.jpg").read_bytes() == b"original-image-bytes"
    assert provenance["archive_sha256"] == checksum
    assert acquisition.acquire_lfw(tmp_path / "raw")[0] == directory


def test_archive_extraction_rejects_escape_and_links(tmp_path):
    from efu.acquisition import extract_tar

    for filename, link in [("../outside", False), ("lfw/link", True)]:
        raw = tmp_path / "bad.tgz"
        with tarfile.open(raw, "w:gz") as archive:
            info = tarfile.TarInfo(filename)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/escape"
            archive.addfile(info)
        with pytest.raises(ValueError):
            extract_tar(raw, tmp_path / "target")
    assert not (tmp_path / "outside").exists()


def test_femnist_raw_preparation_preserves_writers_labels_and_ids(
    tmp_path, monkeypatch
):
    import zipfile
    from PIL import Image
    import numpy as np
    from efu.acquisition import prepare_femnist_archives
    from efu.femnist import FEMNIST

    archives = [tmp_path / "by_class.zip", tmp_path / "by_write.zip"]
    with (
        zipfile.ZipFile(archives[0], "w") as by_class,
        zipfile.ZipFile(archives[1], "w") as by_write,
    ):
        for writer in range(2):
            for index in range(4):
                stream = io.BytesIO()
                Image.new("L", (128, 128), 40 + writer * 30 + index).save(
                    stream, format="PNG"
                )
                raw = stream.getvalue()
                label = "30" if index % 2 == 0 else "41"
                by_class.writestr(f"by_class/{label}/hsf_0/{writer}_{index}.png", raw)
                by_write.writestr(
                    f"by_write/hsf_0/writer{writer}/type/{index}.png", raw
                )
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_femnist_archives(*archives, first, seed=7, test_fraction=0.25)
    prepare_femnist_archives(*archives, second, seed=7, test_fraction=0.25)
    train, test = FEMNIST(first / "train"), FEMNIST(first / "test")
    assert (len(train), len(test)) == (6, 2)
    assert train.users == ["hsf_0/writer0", "hsf_0/writer1"]
    assert set(train.targets) == {0, 10}
    assert not set(train.sample_ids) & set(test.sample_ids)
    assert np.array_equal(train.images, FEMNIST(second / "train").images)
    assert train[0][0].size == (28, 28)

    from efu import acquisition
    from efu.dataset_io import prepare_dataset

    monkeypatch.setattr(
        acquisition,
        "acquire_femnist",
        lambda raw, stage, **options: prepare_femnist_archives(
            *archives, stage, **options
        ),
    )
    root = tmp_path / "prepared"
    manifest = prepare_dataset("femnist", root, split_seed=7, test_fraction=0.25)
    assert (
        prepare_dataset("femnist", root, split_seed=7, test_fraction=0.25) == manifest
    )
    with pytest.raises(ValueError, match="split differs"):
        prepare_dataset("femnist", root, split_seed=8, test_fraction=0.25)
    with pytest.raises(ValueError, match="split differs"):
        prepare_dataset("femnist", root, split_seed=7, test_fraction=0.5)
