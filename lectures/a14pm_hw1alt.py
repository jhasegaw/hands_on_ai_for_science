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
    raise RuntimeError("You need to write this part!")


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
    raise RuntimeError("You need to write this part!")


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
    raise RuntimeError("You need to write this part!")
