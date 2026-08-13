# ----------------------------------------------------------------------------
# UNVERIFIED SOLUTION
#
# Written for the workshop -- this is NOT the course author's original.
# It reproduces the expected output published in the notebook, but it has not
# been reviewed by whoever designed the assignment.  Treat it as a reference
# for a TA, not as an answer key.  See solutions/README.md.
#
# embed_sequences below follows the author's own reference implementation in
# lectures/make_protein_embeddings.py -- the script that generated the shipped
# protein_embeddings.npy -- so it should reproduce that file exactly.  It is
# the one function here NOT checked by running it, because doing so needs the
# ESM-2 model downloaded from HuggingFace.
# ----------------------------------------------------------------------------
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
    out = []

    # eval() disables dropout, and no_grad() stops torch building the graph it
    # would need for backprop.  The model is frozen -- we only read from it --
    # so both save time and memory and neither changes the numbers.
    model.eval()
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            chunk = list(seqs[start:start + batch_size])

            # padding=True stretches every sequence in this batch to the
            # longest one, so they can ride through the model as one tensor.
            batch = tokenizer(chunk, padding=True, return_tensors='pt')

            hidden = model(**batch).last_hidden_state       # (b, L, d)

            # The padded tail positions carry activations too, and averaging
            # them in would drag every short sequence toward the same value.
            # The attention mask is 1 on real positions and 0 on padding, so
            # multiplying zeroes the padding out before the sum.
            mask = batch['attention_mask'].unsqueeze(-1)    # (b, L, 1)
            summed = (hidden * mask).sum(dim=1)             # (b, d)

            # Divide by each sequence's own number of real positions, not by
            # the padded length -- that is what makes this a masked MEAN.
            counts = mask.sum(dim=1)                        # (b, 1)
            out.append((summed / counts).numpy())

    return np.concatenate(out).astype(np.float32)


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
    rows = []
    for seq in seqs:
        # Count each standard amino acid.  Anything else in the string (an X
        # for an unknown residue, say) is simply not counted, which is what
        # "ignore any other characters" asks for.
        counts = np.array([seq.count(aa) for aa in AMINO_ACIDS], dtype=float)

        # Dividing by the total turns counts into frequencies, so that a long
        # protein and a short one with the same makeup get the same row.
        total = counts.sum()
        rows.append(counts / total if total else counts)

    return np.array(rows)


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
    # Identical to a13pm_hw2.linear_probe, deliberately: the same probe reads
    # a protein language model's embeddings and a bag of amino-acid counts
    # alike, which is exactly how you compare two representations fairly.
    model = LogisticRegression(max_iter=2000)
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv)
    return float(scores.mean())
