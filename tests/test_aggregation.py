import numpy as np
import torch
from efu.utils.fe import cluster,fe_secure_aggregate

def test_clustered_unequal_weight_aggregation_matches_fixed_point_oracle():
    states=[{'weight':torch.tensor([-.3,.5,.1,.8])},{'weight':torch.tensor([.2,-.4,.8,.9])}]
    sizes=[2,5]
    clustered=[cluster(s,2,seed=3) for s in states]
    expected=sum(np.rint(z[p]*100)/100*w for (z,p,_,_),w in zip(clustered,sizes))/sum(sizes)
    actual=fe_secure_aggregate(states,sizes,kappa=2,precision=2,seed=3)
    np.testing.assert_allclose(actual['weight'].numpy(),expected,atol=1e-7)
