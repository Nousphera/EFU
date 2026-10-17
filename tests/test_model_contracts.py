"""Paper architecture and input contracts; no external weights are downloaded."""

import pytest
import torch
from efu.train import ImageModel, load_data, main


def test_prepared_inputs_use_the_paper_image_shape(imagefolder_args):
    from efu.train import build_parser

    args = build_parser().parse_args(imagefolder_args)
    train, test, partitions, labels, classes = load_data(args)
    assert train[0][0].shape == (3, 224, 224)
    assert test[0][0].shape == (3, 224, 224)
    assert classes == 3 and len(partitions) == 2


def test_only_paper_models_are_accepted():
    with pytest.raises(SystemExit):
        main(["--model", "unsupported_model", "--rounds", "1", "--method", "fedavg"])


@pytest.mark.parametrize("name", ["resnet18", "vit_b_32"])
def test_paper_model_forward_backward_preserves_full_backbone(name):
    model = ImageModel(name, 10)
    images = torch.rand(2, 3, 224, 224)
    output = model(images)
    assert output.shape == (2, 10) and torch.isfinite(output).all()
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert next(model.parameters()).grad.abs().sum() > 0
    if name == "resnet18":
        assert [len(getattr(model.body, f"layer{i}")) for i in range(1, 5)] == [
            2,
            2,
            2,
            2,
        ]
        assert model.body.fc.in_features == 512
        assert sum(p.numel() for p in model.parameters()) == 11181642
    else:
        assert model.body.patch_size == 32 and len(model.body.encoder.layers) == 12
        assert model.body.hidden_dim == 768 and model.body.heads.head.in_features == 768
        assert sum(p.numel() for p in model.parameters()) == 87462922
