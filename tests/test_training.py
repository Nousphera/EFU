import json
from efu.train import main,partition

def test_two_round_training_writes_real_metrics(tmp_path):
    output=tmp_path/'run.json'
    result=main(['--output',str(output)])
    assert len(result['history'])==2
    assert all(0<=r['accuracy']<=1 for r in result['history'])
    assert json.loads(output.read_text())['configuration']['seed']==2025
    assert result['benchmark_reproduction'] is False

def test_dirichlet_partitions_preserve_disjoint_sample_identity():
    subsets=partition([0]*50+[1]*50,clients=3,alpha=10,seed=7)
    flat=[i for subset in subsets for i in subset]
    assert sorted(flat)==list(range(100)) and len(set(flat))==100


def test_output_refuses_overwrite(tmp_path):
    import pytest
    output=tmp_path/'old.json'
    output.write_text('preserve')
    with pytest.raises(FileExistsError):
        main(['--output',str(output)])
    assert output.read_text()=='preserve'


def test_checkpoint_is_loadable_and_matches_report(tmp_path):
    import torch
    output=tmp_path/'new.json'
    result=main(['--output',str(output),'--method','fedavg','--rounds','1'])
    checkpoint=torch.load(output.with_suffix('.pt'),weights_only=True,map_location='cpu')
    assert checkpoint['configuration']==result['configuration']
    assert checkpoint['model']
    assert result['environment']['torch']


def test_unlearning_requester_is_selected_in_partial_rounds():
    import numpy as np
    from efu.train import select_participants
    for seed in range(20):
        chosen=select_participants(list(range(4)),2,np.random.default_rng(seed),required_client=3)
        assert len(chosen)==2 and len(set(chosen))==2 and 3 in chosen


def test_partial_training_executes_scheduled_request():
    result=main(['--clients','4','--participation','0.5','--forget-client','3'])
    last=result['history'][-1]
    assert last['unlearning_client']==3
    assert 3 in last['participants']
