import numpy as np
import torch
from efu.utils.fe import cluster, fe_secure_aggregate


def test_clustered_unequal_weight_aggregation_matches_fixed_point_oracle():
    states = [
        {"weight": torch.tensor([-0.3, 0.5, 0.1, 0.8])},
        {"weight": torch.tensor([0.2, -0.4, 0.8, 0.9])},
    ]
    sizes = [2, 5]
    clustered = [cluster(s, 2, seed=3) for s in states]
    expected = sum(
        np.rint(z[p] * 100) / 100 * w for (z, p, _, _), w in zip(clustered, sizes)
    ) / sum(sizes)
    actual = fe_secure_aggregate(states, sizes, kappa=2, precision=2, seed=3)
    np.testing.assert_allclose(actual["weight"].numpy(), expected, atol=1e-7)


def test_plaintext_aggregation_weights_float_buffers_and_preserves_counter_dtype():
    from efu.aggregate import fedavg_aggr

    states = [
        {
            "weight": torch.tensor([1.0], dtype=torch.float64),
            "running_mean": torch.tensor([2.0]),
            "num_batches_tracked": torch.tensor(3),
        },
        {
            "weight": torch.tensor([5.0], dtype=torch.float64),
            "running_mean": torch.tensor([6.0]),
            "num_batches_tracked": torch.tensor(7),
        },
    ]
    result = fedavg_aggr(states, [1, 3], 1)[0]
    assert result["weight"].dtype == torch.float64
    assert result["weight"].item() == 4.0
    assert result["running_mean"].item() == 5.0
    assert result["num_batches_tracked"].item() == 7
    assert result["num_batches_tracked"].dtype == torch.int64


def test_plaintext_aggregation_rejects_malformed_cohorts():
    import pytest
    from efu.aggregate import fedavg_aggr

    for states, counts in (
        ([], []),
        ([{"weight": torch.ones(1)}], [0]),
        ([{"weight": torch.ones(1)}, {"other": torch.ones(1)}], [1, 1]),
    ):
        with pytest.raises(ValueError):
            fedavg_aggr(states, counts, 1)
