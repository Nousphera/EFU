"""Request manifests must be random, replayable and scoped to requesting clients."""

import pytest
import torch


def test_sample_requests_are_seeded_samples_not_label_sorted_prefixes():
    from efu.requests import build_requests

    partitions = [list(range(100)), list(range(100, 200))]
    labels = [0] * 50 + [1] * 50 + [0] * 50 + [1] * 50
    forget, retain, classes = build_requests(partitions, labels, [0, 1], 0.2, seed=5)
    assert classes == []
    assert len(forget[0]) == len(forget[1]) == 20
    assert {labels[i] for i in forget[0]} == {0, 1}
    assert {labels[i] for i in forget[1]} == {0, 1}
    assert build_requests(partitions, labels, [0, 1], 0.2, seed=5)[0] == forget
    for cid in (0, 1):
        assert not set(forget[cid]) & set(retain[cid])
        assert sorted(forget[cid] + retain[cid]) == partitions[cid]


def test_class_requests_cover_multiple_classes_without_deleting_other_clients():
    from efu.requests import build_requests

    forget, retain, classes = build_requests(
        [[0, 1, 2], [3, 4, 5]],
        [0, 1, 2, 0, 1, 2],
        [0],
        0.1,
        seed=5,
        forget_classes=[0, 2],
    )
    assert forget == {0: [0, 2]}
    assert retain == [[1], [3, 4, 5]]
    assert classes == [0, 2]


def test_request_selection_rejects_empty_unknown_or_duplicate_clients():
    from efu.requests import build_requests

    for requesters in ([], [2], [0, 0]):
        with pytest.raises(ValueError):
            build_requests([[0], [1]], [0, 1], requesters, 0.1, seed=5)


def test_held_out_split_metrics_count_retained_and_forgotten_classes():
    from efu.experiment import evaluate_class_splits
    from torch.utils.data import DataLoader, TensorDataset

    class Identity(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, x):
            return x

    data = DataLoader(
        TensorDataset(
            torch.tensor([[4.0, 0.0], [3.0, 1.0], [0.0, 4.0]]), torch.tensor([0, 1, 1])
        ),
        batch_size=2,
    )
    result = evaluate_class_splits(Identity(), data, [1])
    assert result["all_test"]["samples"] == 3
    assert result["retained_test"]["accuracy"] == 1.0
    assert result["retained_test"]["samples"] == 1
    assert result["forgotten_test"]["accuracy"] == 0.5
    assert result["forgotten_test"]["samples"] == 2


def test_cli_rejects_false_fedosd_encryption_and_has_no_generated_data_route():
    from efu.train import build_parser, validate_args

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "synthetic"])
    with pytest.raises(ValueError, match="FedOSD|fedosd"):
        validate_args(parser.parse_args(["--unlearn-method", "fedosd"]))


def test_public_numeric_defaults_fit_pretrained_running_variance():
    from efu.train import build_parser, validate_args

    args = build_parser().parse_args([])
    assert args.max_abs == 32.0
    args.precision = 7
    with pytest.raises(ValueError, match="precision"):
        validate_args(args)


def test_task_fisher_source_is_validated_before_training():
    from efu.train import build_parser, validate_args

    args = build_parser().parse_args(
        [
            "--task",
            "identity-gender",
            "--dataset",
            "lfw",
            "--unlearn-method",
            "sga_ewc",
            "--fisher-source",
            "retain",
        ]
    )
    with pytest.raises(ValueError, match="identity Fisher"):
        validate_args(args)
