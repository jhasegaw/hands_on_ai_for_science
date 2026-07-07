import unittest
import submitted
import torch
import torch.nn as nn
from gradescope_utils.autograder_utils.decorators import weight
    
def test_linear(x, in_features, out_features):

    ### Create a Layer ###
    layer = nn.Linear(in_features, out_features)

    ### Create copies of the weight/bias ###
    weight_ = layer.weight.detach().clone().requires_grad_()
    bias_ = layer.bias.detach().clone().requires_grad_()

    ### Create a Clone ###
    x_ = x.detach().clone().requires_grad_()

    ### PyTorch Linear Output ###
    torch_linear_out = layer(x)

    ### Our Linear Layer ###
    custom_linear_out = submitted.LinearFunction.apply(x_, weight_, bias_)

    ### Check Outputs are Equivalent ###
    assert torch.allclose(torch_linear_out, custom_linear_out, rtol=1e-3, atol=1e3)

    ### Backpropagate ###
    upstream_grad = torch.randn_like(torch_linear_out)
    torch_linear_out.backward(upstream_grad)
    custom_linear_out.backward(upstream_grad)

    ### Check Gradients ###
    assert torch.allclose(layer.weight.grad, weight_.grad, rtol=1e-3, atol=1e3)
    assert torch.allclose(layer.bias.grad, bias_.grad, rtol=1e-3, atol=1e3)
    assert torch.allclose(x.grad, x_.grad, rtol=1e-3, atol=1e3)

def test_attention(batch_size, seq_len, embed_dim, attention_mask):

    ### Create input tensors
    Q = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    K = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    V = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    dO = torch.randn(batch_size, seq_len, embed_dim)
    causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    ### Create Clone for Testing ###
    Q_ = Q.detach().clone().requires_grad_(True)
    K_ = K.detach().clone().requires_grad_(True)
    V_ = V.detach().clone().requires_grad_(True)
    dO_ = dO.detach().clone()
    causal_mask_ = causal_mask.detach().clone()
    attention_mask_ = attention_mask.detach().clone()

    ### Our Attention Implementation
    output = submitted.AttentionFunction.apply(Q, K, V, causal_mask, attention_mask)
    
    ### Comparison to torch SDPA
    causal_mask_ = causal_mask_.unsqueeze(0)
    attention_mask_ = attention_mask_.unsqueeze(1).repeat(1,seq_len,1)
    mask = causal_mask_ & attention_mask_
    output_ = torch.nn.functional.scaled_dot_product_attention(Q_, K_, V_, attn_mask=mask)

    ### Check Output ###
    assert torch.allclose(output, output_, rtol=1e-2, atol=1e3)

    ### Check Gradients ###
    output.backward(dO)
    output_.backward(dO_)

    assert torch.allclose(Q.grad, Q_.grad, rtol=1e-3, atol=1e3)
    assert torch.allclose(K.grad, K_.grad, rtol=1e-3, atol=1e3)
    assert torch.allclose(V.grad, V_.grad, rtol=1e-3, atol=1e3)

class TestStep(unittest.TestCase):

    @weight(3)
    def test_build_attention_mask_equal_len(self):

        batch = [[0 for _ in range(10)] for _ in range(20)]

        hyp = submitted.build_attention_mask(batch)
        assert hyp.dtype == torch.bool
        solution = torch.ones(20,10).bool()

        assert torch.equal(hyp, solution)   
    
    @weight(3)
    def test_build_attention_mask_diff_len(self):

        batch = [[0 for _ in range(10)] for _ in range(10)] + [[0 for _ in range(7)] for _ in range(10)]

        hyp = submitted.build_attention_mask(batch)
        assert hyp.dtype == torch.bool
        set1 = torch.ones(10,10).bool()
        set2 = torch.ones(10,10).bool()
        set2[:, -3:] = False
        solution = torch.concatenate([set1, set2], dim=0)
        assert torch.equal(hyp, solution)

    @weight(3)
    def test_batch_samples_equal_len(self):

        pad_token_id = -1

        batch = [[1 for _ in range(10)] for _ in range(20)]

        hyp = submitted.batch_samples(batch, pad_token_id)
        solution = torch.ones(20,10)
        assert torch.equal(hyp, solution)

    @weight(3)
    def test_batch_samples_diff_len(self):

        pad_token_id = -1

        batch = [[1 for _ in range(10)] for _ in range(10)] + [[1 for _ in range(7)] for _ in range(10)]
        set1 = torch.ones(10,10)
        set2 = torch.ones(10,10)
        set2[:, -3:] = -1

        hyp = submitted.batch_samples(batch, pad_token_id)
        solution = torch.concatenate([set1, set2], dim=0)

        assert torch.equal(hyp, solution)

    @weight(3)
    def test_create_causal_mask(self):

        seq_len = 5

        hyp = submitted.create_causal_mask(seq_len)
        assert hyp.dtype == torch.bool
        solution = torch.tensor(
            [[[ True, False, False, False, False],
            [ True,  True, False, False, False],
            [ True,  True,  True, False, False],
            [ True,  True,  True,  True, False],
            [ True,  True,  True,  True,  True]]]
        )
        assert torch.equal(hyp, solution)

    @weight(6)
    def test_linear_function_2d_input(self):
        
        input = torch.randn(2,8, requires_grad=True)

        test_linear(input, in_features=8, out_features=16)

    @weight(6)
    def test_linear_function_3d_input(self):
        
        input = torch.randn(4,2,8, requires_grad=True)
        
        test_linear(input, in_features=8, out_features=16)

    @weight(12)
    def test_attention_no_padding(self):

        batch_size = 32
        seq_len = 64
        embed_dim = 128

        ### Create a dummy attention mask for attending to everything ###
        attention_mask = torch.ones(batch_size, seq_len).bool()

        test_attention(batch_size, seq_len, embed_dim, attention_mask)

    @weight(12)
    def test_attention_padding(self):

        batch_size = 32
        seq_len = 64
        embed_dim = 128

        ### Create a dummy attention mask and add some padding ###
        attention_mask = torch.ones(batch_size, seq_len).bool()
        attention_mask[0, -10:] = False
        attention_mask[4, -6:] = False
        attention_mask[6, -2:] = False
        attention_mask[13, -29:] = False

        test_attention(batch_size, seq_len, embed_dim, attention_mask)

