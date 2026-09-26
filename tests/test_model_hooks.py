import torch
from src.hooks import final_attention


def test_native_forward_and_gradients_match_torch():
    torch.manual_seed(1)
    attn=torch.nn.MultiheadAttention(12,3).double().eval().requires_grad_(False)
    x=torch.randn(5,1,12,dtype=torch.double,requires_grad=True)
    expected=attn(x,x,x,need_weights=False)[0]
    actual,t=final_attention(x,attn,'native')
    torch.testing.assert_close(actual,expected)
    eg=torch.autograd.grad(expected.square().sum(),x,retain_graph=True)[0]
    ag=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    torch.testing.assert_close(ag,eg)
    gv,gq=torch.autograd.grad(actual[0].square().sum(),(t['V'],t['Q']))
    assert gv[:,0].abs().sum()>0 and gq[:,0].abs().sum()>0


def test_full_head_channel_gradient_is_scalar_multiple():
    torch.manual_seed(2)
    attn=torch.nn.MultiheadAttention(12,3).double().eval().requires_grad_(False)
    x=torch.randn(5,1,12,dtype=torch.double,requires_grad=True)
    out,t=final_attention(x,attn,'full')
    gv,ga=torch.autograd.grad(out[0].square().sum(),(t['V'],t['attention_output']))
    torch.testing.assert_close(gv[:,0],t['attention_weights'][:,0,0,None]*ga[:,0])
