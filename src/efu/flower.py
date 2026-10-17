"""Stateful Flower transport: local training, key exchange and authorized aggregation."""

import argparse
import hashlib
import io
import json
import math
import time
from collections import OrderedDict
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
from .centroid_fe import Config, Participant, Aggregator

PROJECT = __package__
_EXPERIMENTS = {}


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _experiment(args):
    from .experiment import Experiment

    key = _json(vars(args))
    if key not in _EXPERIMENTS:
        _EXPERIMENTS.clear()
        _EXPERIMENTS[key] = Experiment(args)
    return _EXPERIMENTS[key]


def _encrypted(args):
    return (
        args.method == "enccluster"
        if PROJECT == "enccluster"
        else args.method == "efu" and args.aggregation == "encrypted"
    )


def _layout(state):
    entries = [
        (key, list(value.shape), str(value.dtype)) for key, value in state.items()
    ]
    return hashlib.sha256(_json(entries).encode()).hexdigest()


def _save(context, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    context.state["private"] = ConfigRecord({"state": buffer.getvalue()})


def _load(context):
    if "private" not in context.state:
        return {"cache": {}, "last_round": 0}
    return torch.load(
        io.BytesIO(context.state["private"]["state"]),
        map_location="cpu",
        weights_only=True,
    )


def _cluster(state, args):
    if PROJECT == "enccluster":
        from .clustering import weight_clustering
        from .filters import (
            encode_mapping,
            serialize_filter,
            deserialize_filter,
            reconstruct_mapping,
        )

        z, p, keys, shapes = weight_clustering(state, args.kappa, args.seed)
        carrier = serialize_filter(encode_mapping(p, args.bpe), args.bpe)
        p = reconstruct_mapping(deserialize_filter(carrier, len(p)), len(p), len(z))
    else:
        from .utils.fe import cluster

        z, p, keys, shapes = cluster(state, args.kappa, args.seed)
        carrier = np.asarray(p, dtype=_mapping_dtype(args.kappa)).tobytes()
    flat = np.rint(z * 10**args.precision)[p] / 10**args.precision
    contribution = {}
    offset = 0
    for key, shape in zip(keys, shapes):
        n = int(np.prod(shape)) if shape else 1
        contribution[key] = torch.as_tensor(
            flat[offset : offset + n].reshape(shape), dtype=state[key].dtype
        )
        offset += n
    contribution.update(
        {key: value for key, value in state.items() if not value.is_floating_point()}
    )
    return torch.as_tensor(z), torch.as_tensor(p), carrier, contribution


def _mapping_dtype(kappa):
    return "<u1" if kappa <= 256 else "<u2" if kappa <= 65536 else "<u4"


def _read_mapping(carrier, cfg, count):
    if PROJECT == "enccluster":
        from .filters import deserialize_filter, reconstruct_mapping

        return reconstruct_mapping(
            deserialize_filter(carrier, cfg.dimension), cfg.dimension, count
        )
    dtype = np.dtype(_mapping_dtype(cfg.kappa))
    if not isinstance(carrier, bytes) or len(carrier) != dtype.itemsize * cfg.dimension:
        raise ValueError("Invalid mapping bytes")
    value = np.frombuffer(carrier, dtype=dtype)
    if np.any(value >= count):
        raise ValueError("Invalid mapping centroid slot")
    return value


def _restore(flat, reference, counters):
    state = OrderedDict()
    offset = 0
    for key, value in reference.items():
        if value.is_floating_point():
            n = value.numel()
            state[key] = torch.as_tensor(
                flat[offset : offset + n].reshape(value.shape), dtype=value.dtype
            )
            offset += n
        else:
            arrays = []
            for item in counters:
                if set(item) != set(
                    k for k, v in reference.items() if not v.is_floating_point()
                ):
                    raise ValueError("Integer buffer layout differs")
                array = torch.as_tensor(item[key], dtype=value.dtype)
                if array.shape != value.shape:
                    raise ValueError("Integer buffer shape differs")
                arrays.append(array)
            state[key] = torch.stack(arrays).amax(0)
    if offset != len(flat):
        raise ValueError("Aggregated vector has wrong length")
    return state


def _needs_contribution(args, rnd):
    return (
        PROJECT == "efu"
        and args.method == "efu"
        and args.unlearn_method == "pgd"
        and rnd == args.unlearn_start - 1
    )


def _observe(cache, state, rnd, weight, args):
    if _needs_contribution(args, rnd):
        from .experiment import observe_contribution

        observe_contribution(cache, state, rnd, weight)


def handle(message, context, experiment=None):
    """No client secret or local floating model is returned in encrypted phases."""
    args = argparse.Namespace(**json.loads(context.run_config["arguments"]))
    torch.set_num_threads(args.torch_threads)
    config = message.content["config"]
    phase = config["phase"]
    rnd = config["round"]
    cid = context.node_config["partition-id"]
    if type(cid) != int or not 0 <= cid < args.clients:
        raise ValueError("Invalid client partition ID")
    if phase == "identify":
        return Message(
            RecordDict({"meta": ConfigRecord({"cid": cid})}), reply_to=message
        )
    if config["cid"] != cid or type(rnd) != int or rnd < 1:
        raise ValueError("Client or round mismatch")
    private = _load(context)
    response = {}
    if phase == "prepare":
        if rnd <= private["last_round"]:
            raise ValueError("Replayed training round")
        if private.get("phase") not in (None, "committed"):
            raise ValueError("Previous round is incomplete")
        exp = experiment or _experiment(args)
        state = message.content["model"].to_torch_state_dict()
        kw = {}
        if PROJECT == "efu":
            kw["request_reference"] = (
                message.content["reference"].to_torch_state_dict()
                if "reference" in message.content
                else None
            )
        updated, count, metrics = exp.client_update(
            cid, state, rnd, private["cache"], **kw
        )
        if type(count) != int or count <= 0:
            raise ValueError("Invalid client sample count")
        if _layout(updated) != _layout(state) or any(
            not torch.isfinite(v).all() for v in updated.values()
        ):
            raise ValueError("Local model is nonfinite or has an inconsistent layout")
        extra = {key: value for key, value in metrics.items() if torch.is_tensor(value)}
        metrics = {key: value for key, value in metrics.items() if key not in extra}
        meta = {
            "cid": cid,
            "count": count,
            "layout": _layout(state),
            "metrics": _json(metrics),
        }
        private.update(
            {
                "round": rnd,
                "phase": "prepared",
                "count": count,
                "local": updated,
                "layout": meta["layout"],
            }
        )
        if _encrypted(args):
            if extra:
                raise ValueError("Encrypted methods cannot return plaintext gradients")
            # Local objective names/values and timing can reveal an update's kind.
            private["local_metrics"] = metrics
            meta["metrics"] = "{}"
            z, p, carrier, contribution = _cluster(updated, args)
            private.update({"z": z, "mapping": p, "carrier": carrier})
            if _needs_contribution(args, rnd):
                private["contribution"] = contribution
            meta.update({"dimension": p.numel(), "centroids": z.numel()})
        else:
            response["model"] = ArrayRecord(updated)
            if extra:
                response["gradients"] = ArrayRecord(extra)
        response["meta"] = ConfigRecord(meta)
    elif phase == "setup":
        if private.get("phase") != "prepared" or private["round"] != rnd:
            raise ValueError("Setup precedes local training")
        cfg = Config.from_bytes(config["protocol"])
        index = config["index"]
        if (
            cfg.round_id != rnd
            or cfg.layout_digest != private["layout"]
            or cfg.dimension != private["mapping"].numel()
            or cfg.sizes[index] != private["count"]
        ):
            raise ValueError("Protocol configuration differs from local contribution")
        if (
            cfg.kappa != args.kappa
            or cfg.precision != args.precision
            or cfg.max_abs != args.max_abs
        ):
            raise ValueError("Protocol parameters differ from experiment")
        participant = Participant(cfg, index)
        private.update({"participant": participant.export_private(), "phase": "setup"})
        response["meta"] = ConfigRecord(
            {"cid": cid, "public_key": participant.public_key()}
        )
    elif phase == "encrypt":
        if private.get("phase") != "setup" or private["round"] != rnd:
            raise ValueError("Encryption precedes setup")
        participant = Participant.from_private(private["participant"])
        participant.set_peers(dict(enumerate(config["peers"])))
        cipher = participant.encrypt(private["z"].numpy(), private["mapping"].numpy())
        private.update(
            {"participant": participant.export_private(), "phase": "encrypted"}
        )
        counters = {
            key: value.tolist()
            for key, value in private["local"].items()
            if not value.is_floating_point()
        }
        response["meta"] = ConfigRecord(
            {
                "cid": cid,
                "ciphertext": cipher,
                "mapping": private["carrier"],
                "counters": _json(counters),
            }
        )
        for key in ("local", "z", "mapping", "carrier"):
            private.pop(key)
    elif phase == "authorize":
        if (
            private.get("phase") not in ("encrypted", "authorizing")
            or private["round"] != rnd
        ):
            raise ValueError("Authorization precedes encryption")
        participant = Participant.from_private(private["participant"])
        share = participant.authorize(config["request"])
        private.update(
            {"participant": participant.export_private(), "phase": "authorizing"}
        )
        response["meta"] = ConfigRecord({"cid": cid, "share": share})
    elif phase == "commit":
        if private.get("round") != rnd:
            raise ValueError("Commit round differs")
        if _encrypted(args):
            if private.get("phase") != "authorizing":
                raise ValueError("Commit precedes authorization")
            participant = Participant.from_private(private["participant"])
            if not participant.authorized_complete:
                raise ValueError("Commit requires complete authorization")
            cfg = participant.config
            weight = cfg.sizes[participant.index] / sum(cfg.sizes)
            contribution = private.get("contribution")
        else:
            if private.get("phase") != "prepared":
                raise ValueError("Invalid plaintext commit")
            weight = config["weight"]
            if not isinstance(weight, (int, float)) or not 0 < weight <= 1:
                raise ValueError("Invalid aggregation weight")
            contribution = private["local"]
        _observe(private["cache"], contribution, rnd, weight, args)
        private = {"cache": private["cache"], "last_round": rnd, "phase": "committed"}
        response["meta"] = ConfigRecord({"cid": cid})
    else:
        raise ValueError("Unknown protocol phase")
    _save(context, private)
    return Message(RecordDict(response), reply_to=message)


def _exchange(grid, nodes, phase, rnd, payload=None, arrays=None, timeout=3600):
    messages = []
    for cid, node in nodes.items():
        data = {"phase": phase, "round": rnd, "cid": cid}
        data.update(payload(cid) if callable(payload) else (payload or {}))
        records = {"config": ConfigRecord(data)}
        records.update(arrays or {})
        messages.append(
            Message(
                RecordDict(records),
                dst_node_id=node,
                message_type="train",
                group_id=str(rnd),
            )
        )
    expected = {m.metadata.dst_node_id: m for m in messages}
    replies = {}
    reverse = {node: cid for cid, node in nodes.items()}
    for reply in grid.send_and_receive(messages, timeout=timeout):
        node = reply.metadata.src_node_id
        if (
            node not in expected
            or node in replies
            or reply.has_error()
            or reply.metadata.group_id != str(rnd)
        ):
            raise RuntimeError("Expected one successful reply from every selected node")
        if (
            "meta" not in reply.content
            or reply.content["meta"].get("cid") != reverse[node]
        ):
            raise ValueError("Reply client identity differs")
        replies[node] = reply
    if set(replies) != set(expected):
        raise RuntimeError("Missing replies from required cohort")
    return [replies[node] for node in nodes.values()]


def run_server(grid, args, experiment=None):
    torch.set_num_threads(args.torch_threads)
    exp = experiment or _experiment(args)
    deadline = time.monotonic() + args.registration_timeout
    while len(nodes := list(grid.get_node_ids())) < args.clients:
        if time.monotonic() > deadline:
            raise TimeoutError("Flower clients did not register in time")
        time.sleep(0.1)
    if len(nodes) != args.clients or len(set(nodes)) != len(nodes):
        raise ValueError("Unexpected Flower node cohort")
    # Node IDs are opaque. Bind them to actual partition IDs before sampling.
    messages = [
        Message(
            RecordDict({"config": ConfigRecord({"phase": "identify", "round": 0})}),
            dst_node_id=node,
            message_type="train",
            group_id="0",
        )
        for node in nodes
    ]
    identities = {}
    for reply in grid.send_and_receive(messages, timeout=args.reply_timeout):
        if reply.has_error() or reply.metadata.src_node_id not in nodes:
            raise RuntimeError("Invalid registration reply")
        cid = reply.content["meta"]["cid"]
        if type(cid) != int or not 0 <= cid < args.clients or cid in identities:
            raise RuntimeError("Duplicate client registration")
        identities[cid] = reply.metadata.src_node_id
    if len(identities) != args.clients or len(set(identities.values())) != args.clients:
        raise RuntimeError("Missing registration replies")
    state = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in exp.model.state_dict().items()
    )
    history = []
    request_reference = None
    for rnd in range(1, args.rounds + 1):
        selected = exp.participants(rnd)
        cohort = {cid: identities[cid] for cid in selected}
        if len(cohort) != len(selected) or len(cohort) < 2:
            raise ValueError("Aggregation requires at least two distinct clients")
        arrays = {"model": ArrayRecord(state)}
        if (
            PROJECT == "efu"
            and args.method == "efu"
            and args.unlearn_method == "fedosd"
        ):
            if rnd == args.unlearn_start:
                request_reference = {key: value.clone() for key, value in state.items()}
            if request_reference is not None:
                arrays["reference"] = ArrayRecord(request_reference)
        previous = state
        ready = _exchange(
            grid, cohort, "prepare", rnd, arrays=arrays, timeout=args.reply_timeout
        )
        counts = [reply.content["meta"]["count"] for reply in ready]
        if any(type(n) != int or n < 1 for n in counts):
            raise ValueError("Invalid sample count")
        if any(reply.content["meta"]["layout"] != _layout(state) for reply in ready):
            raise ValueError("Client model layout differs")
        local_metrics = [
            json.loads(reply.content["meta"]["metrics"]) for reply in ready
        ]
        costs = {}
        if _encrypted(args):
            if any(reply.content.array_records for reply in ready):
                raise ValueError("Encrypted preparation exposed plaintext arrays")
            dimension = sum(v.numel() for v in state.values() if v.is_floating_point())
            if any(reply.content["meta"]["dimension"] != dimension for reply in ready):
                raise ValueError("Client dimension differs")
            cfg = Config.create(
                counts,
                dimension=dimension,
                kappa=args.kappa,
                precision=args.precision,
                max_abs=args.max_abs,
                round_id=rnd,
                layout_digest=_layout(state),
            )
            indices = {cid: i for i, cid in enumerate(selected)}
            setup = _exchange(
                grid,
                cohort,
                "setup",
                rnd,
                lambda cid: {"protocol": cfg.to_bytes(), "index": indices[cid]},
                timeout=args.reply_timeout,
            )
            peers = {
                i: reply.content["meta"]["public_key"] for i, reply in enumerate(setup)
            }
            encrypted = _exchange(
                grid,
                cohort,
                "encrypt",
                rnd,
                {"peers": list(peers.values())},
                timeout=args.reply_timeout,
            )
            if any(reply.content.array_records for reply in setup + encrypted):
                raise ValueError("Encrypted transport exposed arrays")
            messages = [reply.content["meta"]["ciphertext"] for reply in encrypted]
            carriers = [reply.content["meta"]["mapping"] for reply in encrypted]
            maps = [
                _read_mapping(carrier, cfg, ready[i].content["meta"]["centroids"])
                for i, carrier in enumerate(carriers)
            ]
            server = Aggregator(cfg, peers)

            def authorized_chunks():
                for request in server.request_chunks(messages, maps):
                    shares = _exchange(
                        grid,
                        cohort,
                        "authorize",
                        rnd,
                        {"request": request},
                        timeout=args.reply_timeout,
                    )
                    if any(reply.content.array_records for reply in shares):
                        raise ValueError("Authorization exposed arrays")
                    yield request, [reply.content["meta"]["share"] for reply in shares]

            flat = server.aggregate_chunks(messages, authorized_chunks())
            state = _restore(
                flat,
                state,
                [json.loads(reply.content["meta"]["counters"]) for reply in encrypted],
            )
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Aggregated state is nonfinite")
            _exchange(grid, cohort, "commit", rnd, timeout=args.reply_timeout)
            costs = server.last_costs | {"mapping_bytes": sum(map(len, carriers))}
        else:
            states = [reply.content["model"].to_torch_state_dict() for reply in ready]
            for metrics, reply in zip(local_metrics, ready):
                if "gradients" in reply.content:
                    metrics.update(reply.content["gradients"].to_torch_state_dict())
            kwargs = {"metrics": local_metrics} if PROJECT == "efu" else {}
            updated = exp.aggregate_plaintext(
                states, counts, selected, state, rnd, **kwargs
            )
            state = OrderedDict(
                (key, torch.as_tensor(updated[key], dtype=value.dtype))
                for key, value in state.items()
            )
            _exchange(
                grid,
                cohort,
                "commit",
                rnd,
                lambda cid: {"weight": counts[selected.index(cid)] / sum(counts)},
                timeout=args.reply_timeout,
            )
            costs = {
                "model_bytes": sum(
                    reply.content["model"].count_bytes() for reply in ready
                )
            }
        metrics = exp.evaluate(state)
        if any(isinstance(v, float) and not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite evaluation")
        row = {
            "round": rnd,
            "participants": selected,
            "sample_counts": counts,
            "communication": costs,
            **metrics,
        }
        if PROJECT == "efu":
            row.update(
                {
                    "phase": exp.phase(rnd),
                    "unlearning_clients": [
                        cid
                        for cid in selected
                        if exp.is_unlearning(rnd) and cid in exp.forget_by_client
                    ],
                    "local_objectives": [
                        {k: v for k, v in item.items() if not k.startswith("_")}
                        for item in local_metrics
                    ],
                    "parameter_mse": sum(
                        float((state[k].double() - previous[k].double()).square().sum())
                        for k in state
                        if state[k].is_floating_point()
                    )
                    / sum(v.numel() for v in state.values() if v.is_floating_point()),
                }
            )
        history.append(row)
        print(_json(row), flush=True)
    return exp.result(history)
