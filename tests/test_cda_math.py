import torch
from src.cda_clip import attribution_math


def test_hand_computed_equations():
    q=torch.tensor([[1.,0.]])
    k=torch.tensor([[[1.,0.],[0.,1.],[-1.,0.]]])
    v=torch.tensor([[[2.,3.],[5.,7.],[11.,13.]]])
    maps,d=attribution_math(q,k,v,torch.tensor([[1.,2.]]),torch.tensor([[0.,1.]]),q)
    torch.testing.assert_close(d['phi'],torch.tensor([[1.,0.,-1.]]))
    torch.testing.assert_close(d['R'],torch.tensor([[0.,1.,0.]]))
    torch.testing.assert_close(maps['full'],torch.tensor([[8.,19.,0.]]))
    torch.testing.assert_close(maps['gcls'],torch.tensor([[8.,19.,37.]]))
    torch.testing.assert_close(maps['baseline'],torch.tensor([[2.,5.,11.]]))
    # Literal printed formula is scalar, distinct from a map and its positive sum.
    assert d['eq11_literal_scalar'].item()==0


def test_zero_cosine_finite():
    maps,d=attribution_math(torch.zeros(1,2),torch.zeros(1,3,2),torch.ones(1,3,2),
                            torch.zeros(1,2),torch.zeros(1,2),torch.ones(1,2))
    assert all(torch.isfinite(x).all() for x in d.values())
    assert maps['full'].sum()==0
