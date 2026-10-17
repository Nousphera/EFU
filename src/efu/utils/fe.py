"""Full floating-state clustering and authorized centroid aggregation."""

import hashlib
import json
import numpy as np
import torch
from sklearn.cluster import KMeans
from efu.centroid_fe import Config, Participant, Aggregator


def _flatten(state):
    keys = [key for key, value in state.items() if value.is_floating_point()]
    if not keys:
        raise ValueError("No floating tensors to aggregate")
    shapes = [tuple(state[key].shape) for key in keys]
    vector = np.concatenate(
        [state[key].detach().cpu().numpy().reshape(-1) for key in keys]
    ).astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite model state")
    return vector, keys, shapes


def _unflatten(vector, keys, shapes):
    out = {}
    offset = 0
    for key, shape in zip(keys, shapes):
        n = int(np.prod(shape)) if shape else 1
        out[key] = torch.as_tensor(
            np.asarray(vector[offset : offset + n]).reshape(shape), dtype=torch.float32
        )
        offset += n
    if offset != len(vector):
        raise ValueError("Model layout differs from vector size")
    return out


def cluster(state, kappa, seed=0):
    vector, keys, shapes = _flatten(state)
    if type(kappa) != int or kappa < 1:
        raise ValueError("kappa must be positive")
    count = min(kappa, len(np.unique(vector)))
    km = KMeans(n_clusters=count, n_init=4, random_state=seed)
    mapping = km.fit_predict(vector[:, None])
    return km.cluster_centers_.ravel(), mapping.astype(np.int64), keys, shapes


def quantized_contribution(state, kappa, precision, seed=0):
    z, p, keys, shapes = cluster(state, kappa, seed)
    out = _unflatten(np.rint(z * 10**precision)[p] / 10**precision, keys, shapes)
    out.update(
        {
            key: value.detach().cpu().clone()
            for key, value in state.items()
            if not value.is_floating_point()
        }
    )
    return out


def fe_secure_aggregate(
    dicts,
    num_samples,
    kappa=64,
    tag=b"efu",
    precision=3,
    max_abs=32.0,
    seed=0,
    round_id=1,
):
    if not dicts or len(dicts) != len(num_samples):
        raise ValueError("Each client needs a sample count")
    layout = [(k, tuple(v.shape), v.dtype) for k, v in dicts[0].items()]
    if any(
        [(k, tuple(v.shape), v.dtype) for k, v in state.items()] != layout
        for state in dicts
    ):
        raise ValueError("Client model layouts differ")
    clustered = [cluster(state, kappa, seed) for state in dicts]
    keys, shapes = clustered[0][2:]
    digest = hashlib.sha256(
        json.dumps([keys, shapes], separators=(",", ":")).encode()
    ).hexdigest()
    cfg = Config.create(
        num_samples,
        dimension=len(clustered[0][1]),
        kappa=kappa,
        precision=precision,
        max_abs=max_abs,
        round_id=round_id,
        layout_digest=digest,
    )
    clients = [Participant(cfg, i) for i in range(len(dicts))]
    peers = {i: client.public_key() for i, client in enumerate(clients)}
    messages = []
    for client, (z, p, _, _) in zip(clients, clustered):
        client.set_peers(peers)
        messages.append(client.encrypt(z, p))
    server = Aggregator(cfg, peers)
    chunks = (
        (request, [client.authorize(request) for client in clients])
        for request in server.request_chunks(messages, [item[1] for item in clustered])
    )
    result = server.aggregate_chunks(messages, chunks)
    out = _unflatten(result, keys, shapes)
    for key, value in dicts[0].items():
        if not value.is_floating_point():
            out[key] = torch.stack([state[key].cpu() for state in dicts]).amax(0)
    return out
