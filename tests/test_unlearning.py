import torch
from torch.utils.data import DataLoader, TensorDataset
from efu.train import ImageModel
from efu.utils.mas import get_weights_importance


def test_mas_is_elementwise_and_bounded_on_resnet18():
    torch.manual_seed(7)
    model = ImageModel("resnet18", 3)
    data = DataLoader(TensorDataset(torch.rand(2, 3, 224, 224), torch.tensor([0, 1])), batch_size=2)
    importance = get_weights_importance(model, data)
    assert set(importance) == dict(model.named_parameters()).keys()
    for value in importance.values():
        assert value.min() >= 0 and value.max() <= 1
    assert importance["body.conv1.weight"].unique().numel() > 1


def test_adversarial_examples_are_prepared_once_and_reused_on_resnet18():
    from efu.client import prepare_unlearning, local_unlearn
    torch.manual_seed(11)
    model = ImageModel("resnet18", 3)
    data = DataLoader(TensorDataset(torch.rand(2, 3, 224, 224), torch.tensor([0, 1])), batch_size=2)
    prepared = prepare_unlearning(model, data, eps=.1, attack_steps=1)
    originals = [batch[2].clone() for batch in prepared['batches']]
    before = model.body.fc.weight.detach().clone()
    local_unlearn(model, data, num_epochs=2, prepared=prepared)
    assert not torch.equal(before, model.body.fc.weight)
    assert all(torch.equal(a, b[2]) for a, b in zip(originals, prepared['batches']))
    assert all(torch.all(b[1] != b[3]) for b in prepared['batches'])
    assert all(torch.max(torch.linalg.vector_norm((b[2] - b[0]).flatten(1), dim=1)) <= .10001 for b in prepared['batches'])
