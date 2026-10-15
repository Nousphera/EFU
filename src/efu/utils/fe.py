# Modified 2026-10-15: coordinate-labelled decentralized FE integration.

import numpy as np
import torch
from sklearn.cluster import KMeans
from efu.coordinate_fe import setup

_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def _is_weight(name):
    return not any(b in name for b in _BUFFERS)


def _flatten(state):
    keys, shapes, chunks = [], [], []
    for k, v in state.items():
        if _is_weight(k):
            keys.append(k); shapes.append(tuple(v.shape))
            chunks.append(v.detach().cpu().numpy().ravel())
    return np.concatenate(chunks).astype(np.float64), keys, shapes


def _unflatten(vec, keys, shapes):
    out, i = {}, 0
    for k, shape in zip(keys, shapes):
        n = int(np.prod(shape)) if shape else 1
        out[k] = torch.tensor(vec[i:i + n].reshape(shape), dtype=torch.float32); i += n
    return out


def cluster(state, kappa, seed=0):
    """Compress weights into (centroids Z[kappa], mapping P[d], keys, shapes)."""
    vec, keys, shapes = _flatten(state)
    km = KMeans(n_clusters=kappa, n_init=4, random_state=seed)
    P = km.fit_predict(vec.reshape(-1, 1))
    return km.cluster_centers_.ravel().astype(np.float64), P.astype(np.int64), keys, shapes


def fe_secure_aggregate(dicts, num_samples, kappa=64, tag=b"efu", precision=3, max_abs=8.0, seed=0):
    """Cluster then use coordinate-labelled decentralized FE.

    Each client expands centroids to d coordinates and sends d ciphertexts.
    See README.md. BatchNorm buffers are aggregated in plaintext.
    """
    if len(dicts) != len(num_samples):
        raise ValueError("Each client must supply a sample count")
    clustered = [cluster(sd, kappa, seed=seed) for sd in dicts]
    keys, shapes = clustered[0][2:]
    if any(item[2] != keys or item[3] != shapes for item in clustered):
        raise ValueError("Clients must use identical state keys and shapes")
    clients, server = setup(num_samples, dimension=len(clustered[0][1]), precision=precision, max_abs=max_abs)
    messages = [client.encrypt(Z[P], round_id=1) for client,(Z,P,_,_) in zip(clients,clustered)]
    flat = server.aggregate(messages, round_id=1)
    agg = _unflatten(flat, keys, shapes)
    total = sum(num_samples)
    for k,v in dicts[0].items():
        if _is_weight(k):
            continue
        if "num_batches_tracked" in k:
            agg[k] = max(sd[k] for sd in dicts)
        else:
            agg[k] = sum(sd[k].float() * (w / total) for sd,w in zip(dicts,num_samples))
    return agg
