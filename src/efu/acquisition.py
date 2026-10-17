"""Acquisition of original LFW and NIST SD19/LEAF-compatible FEMNIST.

No dataset substitutes or model-generated annotations are used. NIST archives
are processed directly, without extracting hundreds of thousands of raw files.
"""

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image

LFW_URL = "https://ndownloader.figshare.com/files/5976018"
LFW_SHA256 = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
NIST_URLS = {
    name: f"https://s3.amazonaws.com/nist-srd/SD19/{name}.zip"
    for name in ("by_class", "by_write")
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, checksum=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"Archive must not be a symlink: {destination}")
    if destination.is_file():
        if checksum is not None and sha256(destination) != checksum:
            raise ValueError(f"Cached archive hash mismatch: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "EFU-research-data/1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
            expected = response.headers.get("Content-Length")
        if expected is not None and temporary.stat().st_size != int(expected):
            raise ValueError("Incomplete dataset archive download")
        if checksum is not None and sha256(temporary) != checksum:
            raise ValueError("Downloaded dataset archive hash mismatch")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def extract_tar(archive, destination):
    destination = Path(destination)
    with tarfile.open(archive, "r:*") as stream:
        members = stream.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            path = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)


def acquire_lfw(raw_root):
    raw_root = Path(raw_root)
    archive = download(LFW_URL, raw_root / "lfw.tgz", LFW_SHA256)
    # Every preparation extracts from the verified archive, rather than trusting
    # mutable extracted cache contents.
    extracted = raw_root / "lfw-original"
    if extracted.exists():
        shutil.rmtree(extracted)
    extract_tar(archive, extracted)
    source = extracted / "lfw"
    if not source.is_dir():
        raise ValueError("LFW archive lacks its lfw/ identity directory")
    return source, {
        "url": LFW_URL,
        "archive_sha256": LFW_SHA256,
        "variant": "original LFW RGB images",
        "checksum_source": "scikit-learn sklearn/datasets/_lfw.py",
    }


def _members(archive):
    for info in sorted(archive.infolist(), key=lambda entry: entry.filename):
        path = PurePosixPath(info.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or (info.external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise ValueError(f"Unsafe NIST archive member: {info.filename}")
        if not info.is_dir() and info.filename.lower().endswith(".png"):
            yield info


def _label(hexadecimal):
    value = int(hexadecimal, 16)
    if 48 <= value <= 57:
        return value - 48
    if 65 <= value <= 90:
        return value - 55
    if 97 <= value <= 122:
        return value - 61
    raise ValueError(f"Unexpected NIST character code {hexadecimal}")


def prepare_femnist_archives(
    by_class, by_write, stage, *, seed=2025, test_fraction=0.2
):
    """Sorted full-data LEAF mapping, per-writer split, 28x28 grayscale uint8.

    Matching follows LEAF's image-content correspondence between class and
    writer archives. Store memory-mapped arrays to avoid full JSON corpora in RAM.
    """
    if not 0 < test_fraction < 1 or not 0 <= seed < 2**32:
        raise ValueError("FEMNIST requires a uint32 seed and 0<test_fraction<1")
    stage = Path(stage)
    labels = {}
    with zipfile.ZipFile(by_class) as archive:
        for info in _members(archive):
            parts = PurePosixPath(info.filename).parts
            if (
                len(parts) != 4
                or parts[0] != "by_class"
                or "hsf" not in parts[2]
                or "mit" in parts[2]
            ):
                continue
            label = _label(parts[1])
            digest = hashlib.sha256(archive.read(info)).hexdigest()
            if digest in labels and labels[digest] != label:
                raise ValueError("Ambiguous NIST image class correspondence")
            labels[digest] = label
    if not labels:
        raise ValueError("No NIST by_class character images found")
    writers = defaultdict(list)
    with zipfile.ZipFile(by_write) as archive:
        for info in _members(archive):
            parts = PurePosixPath(info.filename).parts
            if len(parts) != 5 or parts[0] != "by_write":
                continue
            digest = hashlib.sha256(archive.read(info)).hexdigest()
            if digest not in labels:
                raise ValueError(
                    f"NIST writer image has no class match: {info.filename}"
                )
            writer = "/".join(parts[1:3])
            writers[writer].append((info.filename, labels[digest]))
        if not writers:
            raise ValueError("No NIST by_write character images found")
        rng = np.random.default_rng(seed)
        splits = {"train": [], "test": []}
        for writer, records in sorted(writers.items()):
            if len(records) < 2:
                raise ValueError(
                    f"Writer has insufficient records for train/test: {writer}"
                )
            order = rng.permutation(len(records))
            count = min(len(records) - 1, max(1, round(len(records) * test_fraction)))
            for split, indices in [("test", order[:count]), ("train", order[count:])]:
                splits[split].extend((writer, *records[i]) for i in sorted(indices))
        for split, records in splits.items():
            folder = stage / split
            folder.mkdir(parents=True)
            images = np.lib.format.open_memmap(
                folder / "images.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(len(records), 28, 28),
            )
            targets = np.lib.format.open_memmap(
                folder / "labels.npy", mode="w+", dtype=np.int64, shape=(len(records),)
            )
            identities = []
            sample_ids = []
            for index, (writer, filename, label) in enumerate(records):
                with Image.open(io.BytesIO(archive.read(filename))) as image:
                    gray = image.convert("L")
                    gray.thumbnail((28, 28), Image.Resampling.LANCZOS)
                    if gray.size != (28, 28):
                        raise ValueError(
                            f"Unexpected NIST image aspect ratio: {filename}"
                        )
                    images[index] = np.asarray(gray)
                targets[index] = label
                identities.append(writer)
                sample_ids.append(filename)
            images.flush()
            targets.flush()
            del images, targets
            (folder / "records.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "writers": identities,
                        "sample_ids": sample_ids,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "implementation": "sorted NIST SD19 content matching; LEAF label/pixel convention; uint8-v1",
        "split_seed": seed,
        "test_fraction": test_fraction,
        "sampling_fraction": 1.0,
        "class_archive_sha256": sha256(by_class),
        "writer_archive_sha256": sha256(by_write),
        "sources": NIST_URLS,
    }


def acquire_femnist(raw_root, stage, *, seed=2025, test_fraction=0.2):
    raw_root = Path(raw_root)
    archives = [
        download(NIST_URLS[name], raw_root / f"{name}.zip")
        for name in ("by_class", "by_write")
    ]
    return prepare_femnist_archives(
        *archives, stage, seed=seed, test_fraction=test_fraction
    )
