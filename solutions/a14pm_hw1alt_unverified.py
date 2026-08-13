# ----------------------------------------------------------------------------
# UNVERIFIED SOLUTION
#
# Written for the workshop -- this is NOT the course author's original.
# It reproduces the expected output published in the notebook, but it has not
# been reviewed by whoever designed the assignment.  Treat it as a reference
# for a TA, not as an answer key.  See solutions/README.md.
#
# embed_images below follows the author's own reference implementation in
# lectures/make_image_embeddings.py -- the script that generated the shipped
# image_embeddings.npy -- so it should reproduce that file exactly.  It is the
# one function here NOT checked by running it, because doing so needs the
# DINOv2 model downloaded from HuggingFace.
# ----------------------------------------------------------------------------
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def embed_images(model, processor, images, batch_size=32):
    '''
    Extract one embedding per image from a pretrained vision model.

    For each batch of images (process in chunks of batch_size):
      1. Preprocess:  batch = processor(images=list_of_images,
         return_tensors='pt')
      2. Forward pass (inside torch.no_grad()):
         hidden = model(**batch).last_hidden_state    # (b, 1+patches, d)
      3. Keep the CLS token -- position 0 -- for each image:
         hidden[:, 0].  (The CLS token is the model's built-in summary of
         the whole image; this is the alternative to the mean-pooling used
         for proteins this morning.)

    @param:
    model - a HuggingFace AutoModel (frozen; never trained here)
    processor - the matching AutoImageProcessor
    images (array or list, each image (H, W, 3) uint8)
    batch_size (int): images per forward pass

    @return:
    embeddings (ndarray of float32, shape (len(images), hidden_size))

    Check: your output should reproduce the shipped image_embeddings.npy
    (generated with exactly this recipe -- see make_image_embeddings.py).
    '''
    out = []

    # Frozen model: eval() turns off dropout, no_grad() skips the autograd
    # bookkeeping we would only need if we were training it.
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = list(images[start:start + batch_size])

            # The processor does the model-specific preparation -- resize,
            # crop, rescale, normalize.  Doing it by hand is where homemade
            # pipelines usually go wrong, so let the matching processor do it.
            batch = processor(images=chunk, return_tensors='pt')

            hidden = model(**batch).last_hidden_state   # (b, 1+patches, d)

            # Position 0 is the CLS token: a slot that attends over every
            # patch, which the model has already trained to summarize the
            # whole image.  Positions 1.. are the individual patches.
            out.append(hidden[:, 0].numpy())

    return np.concatenate(out).astype(np.float32)


def color_histogram(images, bins=8):
    '''
    The deliberately simple alternative representation: each image becomes
    its color distribution.

    For each image, histogram each of the 3 color channels into `bins`
    equal-width bins over the range [0, 256), concatenate the three
    histograms, and divide by the total count -- so each row is a set of
    3*bins fractions summing to 1.

    @param:
    images (array, shape (n, H, W, 3), uint8)
    bins (int): bins per channel

    @return:
    X (ndarray, shape (n, 3*bins)): rows of fractions summing to 1

    Hint: np.histogram(im[:, :, c], bins=bins, range=(0, 256))[0]
    '''
    rows = []
    for im in images:
        # One histogram per colour channel, then laid end to end.  This throws
        # away every bit of spatial structure -- where things are, what shape
        # they have -- and keeps only "how much of each colour".  That is the
        # point: it is the honest simple baseline the embeddings must beat.
        counts = np.concatenate([
            np.histogram(im[:, :, c], bins=bins, range=(0, 256))[0]
            for c in range(3)
        ]).astype(float)

        # Normalising makes the row a set of fractions, so images of different
        # sizes stay comparable.
        rows.append(counts / counts.sum())

    return np.array(rows)


def probe(X, y, groups=None):
    '''
    Measure how much information about y is linearly decodable from a
    frozen representation X, using cross-validated accuracy.

    This is yesterday's probe with ONE upgrade: the features are
    standardized INSIDE each fold, using a Pipeline --
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) --
    so that scaling statistics are always computed from the training rows
    only (the leakage rule from Thursday morning).  cross_val_score
    handles the rest.

    If groups is given, use GroupKFold(5) with those groups; otherwise
    KFold(5, shuffle=True, random_state=0).  Return the mean accuracy.

    @param:
    X (array, shape (n, d)): representations
    y (array, shape (n,)): labels
    groups (array or None): group ids for grouped splitting

    @return:
    acc (float): mean cross-validated accuracy
    '''
    # The Pipeline is what keeps this honest.  Scaling X once up front would
    # compute the mean and standard deviation from every row -- including the
    # fold about to be used for testing -- which is the same leak as selecting
    # features before splitting.  Inside a Pipeline, cross_val_score refits
    # the scaler on each training fold and only then applies it to the test
    # fold.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv)
    return float(scores.mean())
