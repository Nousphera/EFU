"""Native Flower messages with actual scientific updates and test-only small models."""

import json

import pytest
import torch
from flwr.app import Context, RecordDict
from flwr.supercore.task_identity import TaskIdentity
from torch.utils.data import DataLoader, TensorDataset

from efu import flower
from efu.experiment import Experiment, cpu_state, state_digest
from efu.requests import build_requests
from efu.train import build_parser, validate_args


class ClassifierFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 3)
        self.label2class = {i: i for i in range(3)}

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.layer(x.flatten(1))


class TaskFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(4, 4), torch.nn.Tanh()
        )
        self.identity_head = torch.nn.Linear(4, 3)
        self.gender_head = torch.nn.Linear(4, 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        features = self.backbone(x)
        return {
            "identity": self.identity_head(features),
            "gender": self.gender_head(features),
        }


class FixtureExperiment(Experiment):
    """Keep production phase/update/evaluation logic; inject only test data/model."""

    def __init__(self, args):
        self.args = validate_args(args)
        self.task = args.task == "identity-gender"
        torch.manual_seed(11)
        self.model = TaskFixture() if self.task else ClassifierFixture()
        self.classes = 3
        self.partitions = [
            list(range(cid * 4, (cid + 1) * 4)) for cid in range(args.clients)
        ]
        x = torch.rand(args.clients * 4, 1, 2, 2)
        targets = torch.arange(len(x)) % 3
        y = (
            torch.stack([targets, torch.arange(len(x)) % 2], 1)
            if self.task
            else targets
        )
        self.train_data = self.test_data = TensorDataset(x, y)
        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size)
        if self.task:
            self.forget_by_client = dict(enumerate(self.partitions))
            self.retain_by_client = self.partitions
            self.forget_classes = []
        else:
            self.forget_by_client, self.retain_by_client, self.forget_classes = (
                build_requests(
                    self.partitions,
                    targets,
                    [0],
                    args.forget_fraction,
                    seed=args.request_seed,
                )
            )
        self.initial_model_sha256 = state_digest(cpu_state(self.model))
        self.manifest_sha256 = "test-fixture"
        self.annotation_metadata = {"source": "test-fixture"}
        self.request_reference = None


@pytest.fixture(autouse=True)
def task_identity(monkeypatch):
    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)


class Grid:
    def __init__(self, args):
        self.args = args
        self.contexts = {
            n: Context(
                1,
                n,
                {"partition-id": n - 10},
                RecordDict(),
                {"arguments": json.dumps(vars(args))},
            )
            for n in range(10, 10 + args.clients)
        }
        self.wire = []

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, timeout=None):
        replies = []
        for message in messages:
            context = self.contexts[message.metadata.dst_node_id]
            TaskIdentity._node_id = context.node_id
            try:
                # A new experiment each call exercises serialized Context state.
                reply = flower.handle(
                    message, context, experiment=FixtureExperiment(self.args)
                )
            finally:
                TaskIdentity._node_id = 0
            self.wire.append((message, reply))
            replies.append(reply)
        return replies


def arguments(*options):
    args = build_parser().parse_args(
        [
            "--clients",
            "2",
            "--rounds",
            "3",
            "--unlearn-start",
            "2",
            "--unlearn-rounds",
            "1",
            "--unlearn-epochs",
            "1",
            "--attack-steps",
            "1",
            "--batch-size",
            "4",
            "--kappa",
            "3",
            "--forget-fraction",
            ".5",
            "--from-scratch",
            *options,
        ]
    )
    args.reply_timeout, args.registration_timeout, args.torch_threads = 30.0, 1.0, 1
    return args


def test_encrypted_efu_runs_all_phases_without_plaintext_client_arrays():
    args = arguments()
    grid = Grid(args)
    result = flower.run_server(grid, args, experiment=FixtureExperiment(args))
    assert [r["phase"] for r in result["history"]] == [
        "learning",
        "unlearning",
        "retain-learning",
    ]
    assert result["history"][1]["unlearning_clients"] == [0]
    assert result["history"][1]["local_objectives"] == [{}, {}]
    assert result["history"][1]["communication"]["ciphertext_points"] > 0
    assert {message.content["config"]["phase"] for message, _ in grid.wire} == {
        "identify",
        "prepare",
        "setup",
        "encrypt",
        "authorize",
        "commit",
    }
    assert all(not reply.content.array_records for _, reply in grid.wire)
    assert all("private" not in reply.content for _, reply in grid.wire)
    cache = flower._load(grid.contexts[10])["cache"]
    assert "prepared" in cache
    assert all(t.device.type == "cpu" for t in cache["prepared"]["reference"].values())


def test_encrypted_partial_pgd_reference_uses_actual_previous_contribution():
    args = arguments(
        "--clients",
        "3",
        "--participation",
        ".67",
        "--unlearn-method",
        "pgd",
        "--variant",
        "baseline",
    )
    grid = Grid(args)
    result = flower.run_server(grid, args, experiment=FixtureExperiment(args))
    assert all(0 in result["history"][i]["participants"] for i in (0, 1))
    cache = flower._load(grid.contexts[10])["cache"]
    request = next(
        m
        for m, _ in grid.wire
        if m.content["config"]["phase"] == "prepare"
        and m.content["config"]["round"] == 2
        and m.content["config"]["cid"] == 0
    )
    previous_global = request.content["model"].to_torch_state_dict()
    for name, anchor in cache["prepared"]["reference"].items():
        expected = (
            previous_global[name]
            - cache["last_weight"] * cache["last_contribution"][name]
        ) / (1 - cache["last_weight"])
        torch.testing.assert_close(anchor, expected)
    assert cache["last_round"] == 1


def test_fedosd_rejects_duplicate_cohort_and_missing_task_gradient():
    args = arguments(
        "--task",
        "identity-gender",
        "--dataset",
        "lfw",
        "--unlearn-method",
        "fedosd",
        "--aggregation",
        "plaintext",
        "--variant",
        "baseline",
    )
    exp = FixtureExperiment(args)
    state = cpu_state(exp.model)
    with pytest.raises(ValueError, match="unique|cohort"):
        exp.aggregate_plaintext([state, state], [4, 4], [0, 0], state, 2)
    with pytest.raises(ValueError, match="retained gender gradient"):
        exp.aggregate_plaintext([state, state], [4, 4], [0, 1], state, 2)


def test_task_fedosd_transports_retained_gradients_and_fixed_recovery_reference():
    args = arguments(
        "--task",
        "identity-gender",
        "--dataset",
        "lfw",
        "--unlearn-method",
        "fedosd",
        "--aggregation",
        "plaintext",
        "--variant",
        "baseline",
    )
    grid = Grid(args)
    result = flower.run_server(grid, args, experiment=FixtureExperiment(args))
    assert [r["phase"] for r in result["history"]] == [
        "joint-learning",
        "identity-unlearning",
        "gender-only",
    ]
    assert result["history"][1]["parameter_mse"] > 0
    request_messages = [
        (m, r)
        for m, r in grid.wire
        if m.content["config"]["phase"] == "prepare"
        and m.content["config"]["round"] == 2
    ]
    assert len(request_messages) == 2
    for message, reply in request_messages:
        gradient = reply.content["gradients"].to_torch_state_dict()[
            "_retained_gradient"
        ]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        reference = message.content["reference"].to_torch_state_dict()
        cache = flower._load(grid.contexts[message.metadata.dst_node_id])["cache"]
        for name, value in reference.items():
            torch.testing.assert_close(cache["request_reference"][name], value)
    recovery_messages = [
        m
        for m, _ in grid.wire
        if m.content["config"]["phase"] == "prepare"
        and m.content["config"]["round"] == 3
    ]
    first_reference = request_messages[0][0].content["reference"].to_torch_state_dict()
    for message in recovery_messages:
        for name, value in message.content["reference"].to_torch_state_dict().items():
            torch.testing.assert_close(value, first_reference[name])
    assert all(
        "_retained_gradient" not in item
        for row in result["history"]
        for item in row["local_objectives"]
    )


def test_encrypted_transport_rejects_plaintext_gradient_payload():
    args = arguments()
    exp = FixtureExperiment(args)
    original = exp.client_update

    def with_gradient(*a, **kw):
        state, count, metrics = original(*a, **kw)
        return state, count, metrics | {"_retained_gradient": torch.ones(1)}

    exp.client_update = with_gradient
    grid = Grid(args)
    from flwr.app import ArrayRecord, ConfigRecord, Message

    message = Message(
        RecordDict(
            {
                "config": ConfigRecord({"phase": "prepare", "round": 1, "cid": 0}),
                "model": ArrayRecord(cpu_state(exp.model)),
            }
        ),
        dst_node_id=10,
        message_type="train",
        group_id="1",
    )
    TaskIdentity._node_id = 10
    with pytest.raises(ValueError, match="plaintext gradients"):
        flower.handle(message, grid.contexts[10], experiment=exp)


def test_chunked_authorization_keeps_state_until_complete(monkeypatch):
    original = flower.Aggregator.request_chunks
    monkeypatch.setattr(
        flower.Aggregator,
        "request_chunks",
        lambda self, messages, maps: original(self, messages, maps, chunk_size=2),
    )
    args = arguments()
    exp = FixtureExperiment(args)
    grid = Grid(args)
    flower.run_server(grid, args, experiment=exp)
    phases = [m.content["config"]["phase"] for m, _ in grid.wire]
    assert phases.count("authorize") > args.clients * args.rounds
    assert phases.count("commit") == args.clients * args.rounds
    for context in grid.contexts.values():
        private = flower._load(context)
        assert private["phase"] == "committed"
        assert private["last_round"] == args.rounds
        assert "participant" not in private
    assert all(not reply.content.array_records for _, reply in grid.wire)


def test_authorization_does_not_reload_training_cache(monkeypatch):
    args = arguments()
    exp = FixtureExperiment(args)
    grid = Grid(args)
    original_load = torch.load
    touched = []
    original_exchange = grid.send_and_receive
    active_phase = [None]

    def exchange(messages, timeout=None):
        messages = list(messages)
        active_phase[0] = messages[0].content["config"]["phase"]
        return original_exchange(messages, timeout)

    def load(*a, **kw):
        result = original_load(*a, **kw)
        if active_phase[0] in ("authorize", "setup", "encrypt"):
            if (isinstance(result, dict) and "cache" in result) or any(
                "training" in c.state
                and a[0].getvalue() == c.state["training"]["state"]
                for c in grid.contexts.values()
            ):
                touched.append(active_phase[0])
        return result

    grid.send_and_receive = exchange
    monkeypatch.setattr(torch, "load", load)
    flower.run_server(grid, args, experiment=exp)
    assert not touched, "Key exchanges must not deserialize the training cache"
    assert all("cache" in flower._load(c) for c in grid.contexts.values())


def test_cache_encoding_failure_does_not_advance_protocol(monkeypatch):
    args = arguments()
    context = Grid(args).contexts[10]
    flower._save(context, {"cache": {"old": True}, "last_round": 0})
    before = {key: record["state"] for key, record in context.state.items()}
    original = flower._encode_private

    def encode(value):
        if value == {"new": True}:
            raise RuntimeError("Cannot serialize training cache")
        return original(value)

    monkeypatch.setattr(flower, "_encode_private", encode)
    with pytest.raises(RuntimeError, match="training cache"):
        flower._save(
            context, {"cache": {"new": True}, "last_round": 1, "phase": "committed"}
        )
    assert {key: record["state"] for key, record in context.state.items()} == before
