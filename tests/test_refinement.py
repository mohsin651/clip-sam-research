import torch
from src.refinement import spatial_maps


def test_minmax_prior_avoids_signed_weight_inversion():
    phi=torch.linspace(-.9,-.1,196)[None]
    d={'Q_cls':torch.ones(1,2),'K':torch.ones(1,197,2),'V':torch.ones(1,197,2),
       'phi':phi,'R':torch.zeros(1,196),'G_cls':torch.ones(1,2)}
    prior,full=spatial_maps(d)
    torch.testing.assert_close(full,2*((phi-phi.min())/(phi.max()-phi.min())).reshape(14,14))
    torch.testing.assert_close(prior,full)
    d['R']=torch.full((1,196),.25)
    _,full_r=spatial_maps(d)
    torch.testing.assert_close(full_r,full+.5)


def test_constant_prior_is_safe():
    d={'Q_cls':torch.ones(1,2),'K':torch.ones(1,197,2),'V':torch.ones(1,197,2),
       'phi':torch.ones(1,196),'R':torch.zeros(1,196),'G_cls':torch.ones(1,2)}
    a,b=spatial_maps(d)
    assert torch.isfinite(b).all() and b.sum()==0
