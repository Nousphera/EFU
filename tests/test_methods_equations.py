"""Analytic fixtures catch sign, cancellation and projection errors."""

import copy
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class LinearFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0], [-1.0]]))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.label2class = {0: 0, 1: 1}

    @property
    def device(self):
        return self.weight.device

    def forward(self, x):
        return x @ self.weight.T + self.bias


def test_mas_averages_absolute_per_example_gradients_without_cancellation():
    from efu.utils.mas import compute_mas_importance

    model = LinearFixture().train()
    data = TensorDataset(torch.tensor([[-1.0], [1.0]]), torch.tensor([0, 1]))
    before = copy.deepcopy(model.state_dict())
    for size in (1, 2):
        result = compute_mas_importance(model, DataLoader(data, batch_size=size))
        torch.testing.assert_close(result["bias"], torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(result["weight"], torch.tensor([[2.0], [2.0]]))
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_empirical_fisher_is_mean_of_squared_sample_scores():
    from efu.utils.ewc import empirical_fisher

    model = LinearFixture()
    with torch.no_grad():
        model.weight.zero_()
    data = TensorDataset(torch.ones(2, 1), torch.tensor([0, 1]))
    for size in (1, 2):
        result = empirical_fisher(model, DataLoader(data, batch_size=size))
        torch.testing.assert_close(result["bias"], torch.tensor([0.25, 0.25]))
        torch.testing.assert_close(result["weight"], torch.tensor([[0.25], [0.25]]))


def test_fedosd_uses_half_probability_and_real_retained_nullspace():
    from efu.utils.baselines import unlearning_cross_entropy, fedosd_direction

    loss = unlearning_cross_entropy(torch.tensor([[1000.0, 0.0]]), torch.tensor([0]))
    assert loss == pytest.approx(torch.log(torch.tensor(2.0)).item())
    direction = fedosd_direction(
        torch.tensor([3.0, 4.0]), torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    )
    torch.testing.assert_close(direction, torch.tensor([0.0, -5.0]))
    torch.testing.assert_close(
        fedosd_direction(torch.tensor([3.0, 0.0]), torch.tensor([[1.0, 0.0]])),
        torch.zeros(2),
    )
    assert torch.isfinite(fedosd_direction(torch.zeros(2), torch.zeros(1, 2))).all()


def test_fedosd_recovery_prevents_reversion_without_altering_safe_gradient():
    from efu.utils.baselines import project_recovery_gradient

    gradient = project_recovery_gradient(
        torch.tensor([3.0, 4.0]), torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(gradient, torch.tensor([0.0, 5.0]))
    safe = torch.tensor([-3.0, 4.0])
    torch.testing.assert_close(
        project_recovery_gradient(safe, torch.tensor([1.0, 0.0])), safe
    )
    torch.testing.assert_close(project_recovery_gradient(safe, torch.zeros(2)), safe)


def test_pgd_reference_is_weighted_leave_one_out_and_projection_uses_it():
    from efu.utils.baselines import leave_one_out_reference, project_parameters

    global_state = {"weight": torch.tensor([4.0, 8.0])}
    contribution = {"weight": torch.tensor([1.0, 2.0])}
    result = leave_one_out_reference(global_state, contribution, 0.25)
    torch.testing.assert_close(result["weight"], torch.tensor([5.0, 10.0]))
    model = LinearFixture()
    anchor = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    project_parameters(model, anchor, 0.5)
    assert torch.linalg.vector_norm(
        torch.cat([p.flatten() for p in model.parameters()])
    ).item() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        leave_one_out_reference(global_state, contribution, 1.0)


def test_sga_ewc_ascent_and_fixed_anchor_have_expected_directions():
    from efu.utils.baselines import sga_ewc, prepare_sga_ewc

    model = LinearFixture()
    data = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.tensor([0, 0])), batch_size=2
    )
    prepared = prepare_sga_ewc(model, data)
    anchor = {n: v.clone() for n, v in prepared["reference"].items()}
    before = torch.nn.functional.cross_entropy(
        model(torch.ones(2, 1)), torch.tensor([0, 0])
    ).item()
    sga_ewc(model, data, lr=0.1, lambda_ewc=0.0, prepared=prepared)
    after = torch.nn.functional.cross_entropy(
        model(torch.ones(2, 1)), torch.tensor([0, 0])
    ).item()
    assert after > before
    sga_ewc(model, data, lr=0.1, prepared=prepared)
    for name in anchor:
        torch.testing.assert_close(prepared["reference"][name], anchor[name])


def test_best_targeted_attack_keeps_improved_candidate_even_without_success():
    from efu.utils.apgd import apgd

    model = LinearFixture().eval()
    x, target = torch.tensor([[0.8]]), torch.tensor([1])
    torch.manual_seed(4)
    result = apgd(
        model,
        x,
        target,
        eps=0.1,
        norm=2,
        targeted=True,
        n_iter=2,
        best_loss=True,
        loss_function="ce",
    )
    assert model(result).argmax(1).item() != target.item()
    assert result.item() < x.item()
    assert abs(result.item() - x.item()) <= 0.100001


def test_best_loss_attack_honors_requested_iteration_budget():
    from efu.utils.apgd import apgd

    class CountingLinear(LinearFixture):
        calls = 0

        def forward(self, x):
            self.calls += 1
            return super().forward(x)

    model = CountingLinear().eval()
    apgd(
        model,
        torch.tensor([[0.8]]),
        torch.tensor([1]),
        eps=0.1,
        norm=2,
        targeted=True,
        n_iter=2,
        best_loss=True,
        loss_function="ce",
    )
    assert model.calls == 3  # one initialization, two requested gradient steps
