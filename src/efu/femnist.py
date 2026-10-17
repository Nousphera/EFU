"""LEAF FEMNIST 28x28 JSON adapter retaining writer identities and integer labels."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FEMNIST(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.classes = [str(i) for i in range(62)]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        directory = Path(root)
        if (directory / "images.npy").is_file():
            self.images = np.load(
                directory / "images.npy", mmap_mode="r", allow_pickle=False
            )
            self.targets = np.load(
                directory / "labels.npy", mmap_mode="r", allow_pickle=False
            )
            metadata = json.loads((directory / "records.json").read_text())
            writers = metadata.get("writers", [])
            self.sample_ids = metadata.get("sample_ids", [])
            if (
                metadata.get("schema_version") != 1
                or self.images.dtype != np.uint8
                or self.images.shape != (len(self.targets), 28, 28)
                or len(self.targets) == 0
                or self.targets.dtype.kind not in "iu"
                or self.targets.min() < 0
                or self.targets.max() > 61
                or len(writers) != len(self.targets)
                or any(not isinstance(w, str) for w in writers)
                or len(self.sample_ids) != len(self.targets)
                or any(not isinstance(key, str) for key in self.sample_ids)
                or len(set(self.sample_ids)) != len(self.targets)
            ):
                raise ValueError("Invalid prepared FEMNIST arrays or sample identities")
            self.users = sorted(set(writers))
            groups = {user: [] for user in self.users}
            for index, user in enumerate(writers):
                groups[user].append(index)
            self.user_indices = [groups[user] for user in self.users]
            return
        records = {}
        for path in sorted(Path(root).glob("*.json")):
            data = json.loads(path.read_text())
            users = data["users"]
            if len(set(users)) != len(users) or set(users) != set(data["user_data"]):
                raise ValueError("LEAF users must uniquely match user_data")
            sizes = data.get("num_samples")
            if sizes is not None and len(sizes) != len(users):
                raise ValueError("LEAF num_samples length differs from users")
            for index, user in enumerate(users):
                if not isinstance(user, str) or user in records:
                    raise ValueError(
                        "Writer identities must be strings and unique across shards"
                    )
                entry = data["user_data"][user]
                x = np.asarray(entry["x"], dtype=np.float32)
                y = np.asarray(entry["y"])
                if (
                    x.ndim != 2
                    or x.shape[1] != 784
                    or len(x) == 0
                    or not np.isfinite(x).all()
                    or x.min() < 0
                    or x.max() > 1
                ):
                    raise ValueError(
                        "FEMNIST expects nonempty finite [0,1] vectors of length 784"
                    )
                if (
                    y.shape != (len(x),)
                    or y.dtype.kind not in "iu"
                    or y.min() < 0
                    or y.max() > 61
                ):
                    raise ValueError(
                        "FEMNIST labels must be integers in 0..61, one per image"
                    )
                if sizes is not None and sizes[index] != len(x):
                    raise ValueError("LEAF num_samples differs from actual records")
                records[user] = (x, y.astype(np.int64))
        if not records:
            raise ValueError("No LEAF FEMNIST JSON shards found")
        self.users = sorted(records)
        self.user_indices = []
        offset = 0
        for user in self.users:
            count = len(records[user][0])
            self.user_indices.append(list(range(offset, offset + count)))
            offset += count
        self.images = np.concatenate([records[user][0] for user in self.users])
        self.targets = np.concatenate([records[user][1] for user in self.users])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # LEAF pixels are [0,1]. Convert to RGB for the existing torchvision models.
        pixels = (
            self.images[index]
            if self.images.dtype == np.uint8
            else np.rint(self.images[index].reshape(28, 28) * 255).astype(np.uint8)
        )
        image = Image.fromarray(pixels).convert("RGB")
        return (self.transform(image) if self.transform else image), int(
            self.targets[index]
        )
