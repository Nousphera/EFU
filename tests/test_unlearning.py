import torch
from torch.utils.data import DataLoader, TensorDataset
from efu.utils.mas import get_weights_importance

class Tiny(torch.nn.Linear):
    @property
    def device(self):
        return self.weight.device

def test_mas_is_elementwise_and_bounded():
    model = Tiny(2,2,bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))
    data = DataLoader(TensorDataset(torch.tensor([[1., 0.], [0., 2.]]), torch.zeros(2)), batch_size=1)
    importance = get_weights_importance(model,data)["weight"]
    assert importance.min() >= 0 and importance.max() <= 1
    assert importance.unique().numel() > 1

def test_adversarial_examples_are_prepared_once_and_reused():
    from efu.client import prepare_unlearning, local_unlearn
    model = Tiny(4,3)
    model.label2class = {0:0,1:1,2:2}
    x = torch.rand(6,4)
    y = torch.tensor([0,1,2,0,1,2])
    data = DataLoader(TensorDataset(x,y),batch_size=3)
    prepared = prepare_unlearning(model,data,eps=.1,attack_steps=2)
    originals = [batch[2].clone() for batch in prepared['batches']]
    before = model.weight.detach().clone()
    local_unlearn(model,data,num_epochs=2,prepared=prepared)
    assert not torch.equal(before,model.weight)
    assert all(torch.equal(a,b[2]) for a,b in zip(originals,prepared['batches']))
    assert all(torch.all(b[1] != b[3]) for b in prepared['batches'])
    assert all(torch.max(torch.linalg.vector_norm(b[2]-b[0],dim=1)) <= .10001 for b in prepared['batches'])
