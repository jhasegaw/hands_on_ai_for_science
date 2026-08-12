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
    # Every row of the mask has to be as wide as the longest sequence, since
    # the padded batch will be a rectangle of that width.
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
    # A tensor has to be rectangular, so every sequence gets stretched to the
    # length of the longest one in this batch.
    max_seq_len = max(len(sequence) for sequence in batch)

    # Pre-fill the whole rectangle with the pad token, then write each real
    # sequence over the front of its row.  Whatever is left at the end of a
    # short row is already padding, so there is nothing more to do.
    #
    # dtype=torch.long matters: token IDs are used to index the embedding
    # table, and PyTorch requires an integer type for that.
    data = torch.full((len(batch), max_seq_len), pad_token_id, dtype=torch.long)

    for row, sequence in enumerate(batch):
        data[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)

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
    # Row i of this mask says which positions token i is allowed to look at.
    # Keeping only the lower triangle means token i sees positions 0..i --
    # itself and everything before it, but nothing in the future.  That is
    # what makes the model autoregressive: when predicting token 5 it cannot
    # cheat by reading token 6.
    #
    # torch.tril zeroes everything above the diagonal.  diagonal=0 (the
    # default) keeps the diagonal itself, so each token can attend to itself.
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    # The leading dimension of size 1 is a batch axis.  It lets the same mask
    # broadcast across every sequence in the batch without copying it.
    return mask.unsqueeze(0)
