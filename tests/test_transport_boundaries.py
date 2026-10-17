import argparse
import json
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.supercore.task_identity import TaskIdentity
from efu import flower, launcher


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_launcher_rejects_nonfinite_gpu_resources(value, tmp_path):
    args = launcher.parser().parse_args(
        ["--client-gpus=" + value, "--output", str(tmp_path / "new.json")]
    )
    with pytest.raises(ValueError, match="finite"):
        launcher.configuration(args)


@pytest.mark.parametrize("suffix", [".json", ".pt", ".runtime.log"])
def test_launcher_rejects_existing_artifact_before_loading_data(
    suffix, tmp_path, monkeypatch
):
    import efu.experiment

    path = tmp_path / "run.json"
    existing = path.with_suffix(suffix)
    existing.write_text("prior run")
    monkeypatch.setattr(
        efu.experiment,
        "Experiment",
        lambda *_: pytest.fail("Data loaded before output preflight"),
    )
    with pytest.raises(SystemExit) as error:
        launcher.main(["--output", str(path)])
    assert error.value.code == 2
    assert existing.read_text() == "prior run"


@pytest.mark.parametrize("encrypted", [True, False])
def test_prepare_keeps_sensitive_metrics_local_for_encrypted_methods(
    encrypted, monkeypatch
):
    for key, value in [("_run_id", 1), ("_task_id", 1), ("_node_id", 0)]:
        monkeypatch.setattr(TaskIdentity, key, value)
    args = argparse.Namespace(
        method="efu" if encrypted else "fedavg",
        aggregation="encrypted",
        clients=2,
        torch_threads=1,
        kappa=2,
        bpe=16,
        precision=3,
        seed=42,
        unlearn_method="efu",
        unlearn_start=2,
    )
    context = Context(
        1, 10, {"partition-id": 0}, RecordDict(), {"arguments": json.dumps(vars(args))}
    )
    state = {"weight": torch.tensor([0.1, 0.2, 0.3, 0.4])}
    metrics = {"forget_objective": 1.25, "training_seconds": 0.75}

    class Experiment:
        def client_update(self, *a, **kw):
            return state, 4, metrics

    message = Message(
        RecordDict(
            {
                "config": ConfigRecord({"phase": "prepare", "round": 1, "cid": 0}),
                "model": ArrayRecord(state),
            }
        ),
        dst_node_id=10,
        message_type="train",
        group_id="1",
    )
    TaskIdentity._node_id = 10
    reply = flower.handle(message, context, experiment=Experiment())
    assert json.loads(reply.content["meta"]["metrics"]) == (
        {} if encrypted else metrics
    )
    if encrypted:
        assert flower._load(context)["local_metrics"] == metrics
        assert not reply.content.array_records
