'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise NotImplementedError.  You should replace
each "raise NotImplementedError" line with a line that performs the function specified in the
function's docstring.

This homework is (roughly) the first quarter of a homework assignment created by Priyam Mazumdar
for CS 440/ECE 448 in Spring 2026.

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
    #raise NotImplementedError
    max_seq_len = max(len(sequence) for sequence in batch)

    # Start with everything False ("do not attend"), then switch on the
    # positions that hold real tokens.  Building the mask from the lengths
    # rather than from the padded tensor means we never have to guess whether
    # a token that happens to equal pad_token_id is real padding or not.
    attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)

    for row, sequence in enumerate(batch):
        # The first len(sequence) positions of this row are real tokens; the
        # rest stay False.  A sequence that is already full length simply
        # turns the whole row True.
        attention_mask[row, :len(sequence)] = True


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
    #raise NotImplementedError
    max_seq_len = max(len(sequence) for sequence in batch)
    data = torch.full((len(batch), max_seq_len), pad_token_id, dtype=torch.long)
    for rownumber, sequence in enumerate(batch):
        for columnnumber, item in enumerate(sequence):
            data[rownumber, columnnumber] = item

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
    raise NotImplementedError
    return mask

