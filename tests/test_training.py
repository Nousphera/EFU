import io
import json
import pytest
import torch
from efu.train import main, partition


def test_two_round_training_writes_real_metrics(tmp_path, imagefolder_args):
    output = tmp_path / "run.json"
    result = main(imagefolder_args + ["--output", str(output), "--method", "fedavg"])
    assert len(result["history"]) == 2
    assert all(0 <= r["accuracy"] <= 1 for r in result["history"])
    assert json.loads(output.read_text())["configuration"]["seed"] == 2025
    assert result["benchmark_reproduction"] is False


def test_dirichlet_partitions_preserve_disjoint_sample_identity():
    subsets = partition([0] * 50 + [1] * 50, clients=3, alpha=10, seed=7)
    flat = [i for subset in subsets for i in subset]
    assert sorted(flat) == list(range(100)) and len(set(flat)) == 100


def test_partition_and_request_seeds_are_independent_of_initialization(
    imagefolder_args,
):
    from efu.experiment import Experiment
    from efu.train import build_parser

    parser = build_parser()
    first = Experiment(
        parser.parse_args(imagefolder_args + ["--partition-seed", "17", "--seed", "1"])
    )
    second = Experiment(
        parser.parse_args(imagefolder_args + ["--partition-seed", "17", "--seed", "2"])
    )
    assert first.partitions == second.partitions
    assert first.forget_by_client == second.forget_by_client
    assert first.initial_model_sha256 != second.initial_model_sha256


def test_output_refuses_overwrite(tmp_path):
    output = tmp_path / "old.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert output.read_text() == "preserve"


def test_checkpoint_is_loadable_and_matches_report(tmp_path, imagefolder_args):
    output = tmp_path / "new.json"
    result = main(
        imagefolder_args
        + ["--output", str(output), "--method", "fedavg", "--rounds", "1"]
    )
    checkpoint = torch.load(
        output.with_suffix(".pt"), weights_only=True, map_location="cpu"
    )
    assert checkpoint["configuration"] == result["configuration"]
    assert checkpoint["model"]["body.conv1.weight"].shape == (64, 3, 7, 7)
    assert result["environment"]["torch"]
    assert checkpoint["initial_model_sha256"] == result["initial_model_sha256"]


def test_partial_training_executes_required_request(imagefolder_args):
    result = main(
        imagefolder_args
        + [
            "--clients",
            "4",
            "--participation",
            ".5",
            "--forget-client",
            "3",
            "--request-participation",
            "required",
        ]
    )
    last = result["history"][-1]
    assert last["unlearning_clients"] == [3]
    assert 3 in last["participants"]


@pytest.mark.parametrize("method", ["pgd", "sga_ewc", "fedosd"])
def test_baseline_rounds_and_recovery_run_with_real_image_backbone(
    imagefolder_args, method
):
    result = main(
        imagefolder_args
        + ["--unlearn-method", method, "--variant", "baseline", "--rounds", "3"]
    )
    assert [row["phase"] for row in result["history"]] == [
        "learning",
        "unlearning",
        "retain-learning",
    ]
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in result["history"])
    assert (
        result["history"][2]["sample_counts"][0]
        < result["history"][0]["sample_counts"][0]
    )


def test_pgd_preparation_survives_weights_only_cache_reload(imagefolder_args):
    from efu.train import build_parser
    from efu.experiment import Experiment, cpu_state

    args = build_parser().parse_args(
        imagefolder_args + ["--unlearn-method", "pgd", "--variant", "baseline"]
    )
    exp = Experiment(args)
    state = cpu_state(exp.model)
    caches = {cid: {} for cid in range(args.clients)}
    updates = [
        exp.client_update(cid, state, 1, caches[cid]) for cid in exp.participants(1)
    ]
    states, counts, metrics = map(list, zip(*updates))
    global_state = exp.aggregate_plaintext(
        states, counts, exp.participants(1), state, 1, metrics=metrics
    )
    buf = io.BytesIO()
    torch.save(caches[0], buf)
    buf.seek(0)
    restored = torch.load(buf, weights_only=True)
    result, _, _ = exp.client_update(0, global_state, 2, restored)
    anchor = restored["prepared"]["reference"]["body.fc.weight"]
    expected = (
        global_state["body.fc.weight"]
        - counts[0] / sum(counts) * states[0]["body.fc.weight"]
    ) / (1 - counts[0] / sum(counts))
    torch.testing.assert_close(anchor, expected)
    assert torch.isfinite(result["body.fc.weight"]).all()
