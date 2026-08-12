import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def embed_sequences(model, tokenizer, seqs, batch_size=16):
    '''
    Extract one embedding per sequence from a pretrained protein language
    model, by masked mean-pooling of the last hidden states.

    For each batch of sequences (process seqs in chunks of batch_size):
      1. Tokenize:  batch = tokenizer(list_of_seqs, padding=True,
         return_tensors='pt')
      2. Forward pass (inside torch.no_grad()):
         hidden = model(**batch).last_hidden_state      # (b, L, d)
      3. Mean-pool over REAL positions only: use batch['attention_mask']
         so that padding positions do not contribute to the average --
         sum the masked hidden states over the length dimension and divide
         by the number of real positions in each sequence.

    @param:
    model - a HuggingFace AutoModel (frozen; never trained here)
    tokenizer - the matching AutoTokenizer
    seqs (list of str): amino-acid sequences
    batch_size (int): sequences per forward pass

    @return:
    embeddings (ndarray of float32, shape (len(seqs), hidden_size))

    Check: your output should reproduce the shipped protein_embeddings.npy
    (which was generated with exactly this recipe -- see
    make_protein_embeddings.py).
    '''
    raise RuntimeError("You need to write this part!")


def composition_baseline(seqs):
    '''
    The deliberately simple alternative representation: each sequence
    becomes its amino-acid composition.

    For each sequence, count how many times each of the 20 standard amino
    acids (the string AMINO_ACIDS above, in that order) appears, and
    divide by the total count of those letters -- so each row is a set of
    20 frequencies summing to 1.  Ignore any other characters.

    @param:
    seqs (list of str): amino-acid sequences

    @return:
    X (ndarray, shape (len(seqs), 20)): rows of frequencies, in
      AMINO_ACIDS order
    '''
    raise RuntimeError("You need to write this part!")


def probe(X, y, groups=None):
    '''
    Measure how much information about y is linearly decodable from a
    frozen representation X, using cross-validated accuracy.

    This is the SAME function as linear_probe from yesterday afternoon's
    homework (a13pm_hw2.py) -- reuse your solution.  That is the point:
    one probe, any representation.

    Use LogisticRegression(max_iter=2000).  If groups is given, use
    GroupKFold(5) with those groups; otherwise
    KFold(5, shuffle=True, random_state=0).  Return the mean accuracy
    (cross_val_score).

    @param:
    X (array, shape (n, d)): representations
    y (array, shape (n,)): labels
    groups (array or None): group ids for grouped splitting

    @return:
    acc (float): mean cross-validated accuracy
    '''
    raise RuntimeError("You need to write this part!")
