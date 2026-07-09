'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise NotImplementedError.  You should replace
each "raise NotImplementedError" line with a line that performs the function specified in the
function's docstring.

'''

import torch
import math

def build_attention_mask(batch):
    """
    Create attention mask for padded sequences.
    
        Example: [[1,2,3,4], [1,2,3,4,1,2], [2,3,4]] 
        
        Output: [T T T T F F]
                [T T T T T T]
                [T T T F F F]

    Args:
        batch: List of token ID sequences (where each sequence is also a list)
               where each sequence is of variable length
    
    Returns:
        Boolean torch.Tensor mask of shape (batch_size, max_seq_len)
        where True indicates positions that participate in Attention
        and False indicates positions that do not (pad tokens)
    """
    batch_size = len(batch)
    max_seq_len = max([ len(b) for b in batch ])
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    for row in range(batch_size):
        for token in range(len(batch[row])):
            attention_mask[row,token] = True
    #raise NotImplementedError

    return attention_mask

def batch_samples(batch, pad_token_id):
    """
    Pad a batch of variable-length sequences to the same length.

        Example: [[1,2,3,4], [1,2,3,4,1,2], [2,3,4]] 
        
        Output: [1 2 3 4 P P]       where P is pad token id
                [1 2 3 4 1 2]
                [2 3 4 P P P]
    
    Args:
        batch: List of token ID sequences (where each sequence is also a list)
        pad_token_id: Integer ID to use for padding
    
    Returns:
        Padded torch.Tensor of shape (batch_size, max_seq_len)
    """
    batch_size = len(batch)
    max_seq_len = max([ len(b) for b in batch ])
    data = torch.zeros((batch_size, max_seq_len))
    for row in range(batch_size):
        for col in range(max_seq_len):
            if col < len(batch[row]):
                data[row,col] = batch[row][col]
            else:
                data[row,col] = pad_token_id
    #raise NotImplementedError
    
    return data

def create_causal_mask(seq_len):
    """
    Create causal attention mask (lower triangular).

        Example: 4
        Output: [T F F F]
                [T T F F]
                [T T T F]
                [T T T T]
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Boolean mask of shape (1, seq_len, seq_len)
    """
    alltrue = torch.ones((seq_len,seq_len), dtype=torch.bool)
    mask = torch.tril(alltrue)
    #raise NotImplementedError
    return mask

class LinearFunction(torch.autograd.Function):

    """
    In PyTorch we can use torch.autograd.Function to define forward and custom
    backward methods for an operation. We will be doing this for the Linear layer
    for this problem!
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        input:  (*, D_in) - any number of leading dimensions indicated by *
        weight: (D_out, D_in)
        bias:   (D_out,)
        
        output: (*, D_out)
        """
        ### Save tensors for backward
        ctx.save_for_backward(input, weight, bias)

        raise NotImplementedError
        output = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: dL/dy (*, D_out)
        """
        input, weight, bias = ctx.saved_tensors

        raise NotImplementedError
    
        # dL/dx
        grad_input = 0
        assert grad_input.shape == input.shape

        # dL/dW: 
        grad_weight = 0
        assert grad_weight.shape == weight.shape

        # dL/db:
        grad_bias = 0
        assert grad_bias.shape == bias.shape

        return grad_input, grad_weight, grad_bias


class AttentionFunction(torch.autograd.Function):

    """
    Implementation of the Attention Mechanism:

    Attention(Q,K,V) = Softmax(Q@K^T / sqrt(embed_dim)) @ V
    """
    @staticmethod
    def forward(ctx, Q, K, V, causal_mask, attention_mask):
        """
        Forward pass for single-headed attention.
        
        Args:
            Q: Query tensor of shape (batch, seq_len, embed_dim)
            K: Key tensor of shape (batch, seq_len, embed_dim)
            V: Value tensor of shape (batch, seq_len, embed_dim)
            causal_mask: Boolean tensor of shape (1, seq_len, seq_len) - True for elements that take part in attention/False for elements that are masked
            attention_mask: Boolean tensor of shape (batch, seq_len) - True for elements that take part in attention/False for elements that are masked
        
        Returns:
            output: Attention output of shape (batch, seq_len, d_v)
        """

        batch_size, seq_len, embed_dim = Q.shape # Keys/Values have the same shape as the queries
        
        raise NotImplementedError
    
        ### Compute attention scores: QK^T / sqrt(d_k) -> # (batch, seq_len, seq_len)
        scores = 0
        
        ### Apply causal mask which is in the shape (1, seq_len, seq_len)
        ### The extra 1 at the start will broadcast over the batch dimension
        ### You should be able to use masked_fill to make this easy!
        if causal_mask is not None:

            ### HINT: Our causal mask is True where we want to attend, False where we dont want to. 
            ### But we want to fill in in the False positions, masked_fill will fill in the True positions
            ### by default. So you have to flip the mask here!

            ### Flipping in pytorch is easy, if you have the mask causal_mask, you can flip the boolean by doing ~causal_mask
            scores = 0
        
        ### Apply attention mask for padding tokens which is in the shape (batch, seq_len)
        ### But our attention scores is in the shape of (batch, seq_len, seq_len)
        ### So we need to add an extra dimension to (batch, 1, seq_len)
        ### as in our (seq_len x seq_len) the first seq_len dim refers to the query positions
        ### and the second seq_len dimension refers to the key positions. We want to make sure
        ### no query attends to padding positions in the keys.
        ### And then we will expand the (batch, 1, seq_len) to the (batch, seq_len, seq_len) that we want

        ### So if our mask (assuming a batch_size of 1) is [True, True, False, False]
        ### We want to make sure all query tokens dont attend to the second two keys. 
        ### So in our attention matrix
        ### [A00, A01, A02, A03]
        ### [A10, A11, A12, A13]
        ### [A20, A21, A22, A23]
        ### [A30, A31, A32, A33]

        ### We will make out the columns that refer to the second two positions of the keys
        ### [A00, A01,  X,  X]
        ### [A10, A11,  X,  X]
        ### [A20, A21,  X,  X]
        ### [A30, A31,  X,  X]

        if attention_mask is not None:

            ### Reshape attention_mask to match scores shape (batch, 1, seq_len)
            attention_mask = 0

            ### Expand attention masks to (batch, seq_len, seq_len)
            attention_mask = 0

            ### Apply the Mask (same idea as before as we need to flip the mask)
            scores = 0
        
        ### Softmax to get attention weights (batch, seq_len, seq_len)
        attn_weights = 0
        
        ### Compute output: attention_weights @ V -> (batch, seq_len, d_v)
        output = 0

        ### DONT TOUCH ANYTHING ELSE HERE ###
        ### Save tensors for backward pass ###
        ### We need Q,K,V and the intermediate attn_weights to have all the pieces
        ### for the backward pass
        ctx.save_for_backward(Q, K, V, attn_weights)
        
        ### We also store masks as if something was masked in the attention operation in 
        ### the forward pass, we also need to mask those gradients in the backward pass
        ### as gradients from future positions (or padding positions) should not be included
        ctx.causal_mask = causal_mask
        ctx.attention_mask = attention_mask

        ### We also store the embedding dimension for the scaling factor
        ctx.embed_dim = embed_dim
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for single-headed attention.
        
        Args:
            grad_output: Gradient of loss w.r.t. output, shape (batch, seq_len, d_v)
        
        Returns:
            Gradients w.r.t. Q, K, V, causal_mask, attention_mask

        In this setup, causal_mask and attention_mask have no gradients so we just return None for
        those. Its just in the PyTorch Function method, every input to the forward has to have a return
        in the gradient!

        KEY HINT: Gradients must be in the same shape as the original data!!
        """

        ### Grab all the information we stored in ctx from the forward pass
        Q, K, V, attn_weights = ctx.saved_tensors
        causal_mask = ctx.causal_mask
        attention_mask = ctx.attention_mask
        embed_dim = ctx.embed_dim

        raise NotImplementedError
        
        ### Gradient w.r.t. V: (batch, seq_len, embed_dim)
        grad_V = 0
        
        ### Gradient w.r.t. attn_weights: (batch, seq_len, seq_len)
        grad_attn_weights = 0
        
        ### Gradient through softmax: (batch, seq_len, seq_len)
        grad_scores = 0
        
        ### Apply masks to gradients. We will fill our gradients with 0, as masked positions
        ### Should have no contribution to the final gradient! (You will have to flip the masks
        ### here again as we saved the non-flipped versions in the ctx)
        if causal_mask is not None:
            grad_scores = 0
        if attention_mask is not None:
            grad_scores = 0

        ### Account for scaling by 1/sqrt(d_k) in the backward pass
        grad_scores = 0
        
        ### Gradient w.r.t. Q: (batch, seq_len, embed_dim)
        grad_Q = 0
        
        # Gradient w.r.t. K: (batch, seq_len, embed_dim)
        grad_K = 0
        
        ### DONT TOUCH ANYTHING ELSE HERE ###
        ### Masks don't require gradients
        return grad_Q, grad_K, grad_V, None, None