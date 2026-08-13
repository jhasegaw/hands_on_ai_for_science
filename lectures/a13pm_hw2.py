import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score


def nearest_neighbors(embeddings, query, k=5):
    '''
    Find the k nearest neighbors of one row of an embedding matrix,
    by cosine distance.

    @param:
    embeddings (array, shape (n, d)): one embedding per row
    query (int): the row index whose neighbors are wanted
    k (int): how many neighbors to return

    @return:
    idx (array of int, length k): row indices of the k nearest rows,
      nearest first, NOT including the query row itself

    Hints: scipy.spatial.distance.pdist(embeddings, 'cosine') gives all
    pairwise distances; squareform turns them into an (n, n) matrix;
    np.argsort orders them.  Remember to remove the query index.
    '''
    raise NotImplementedError("You need to write this part!")


def project_2d(embeddings, method="pca"):
    '''
    Project an embedding matrix down to 2 dimensions for plotting.

    @param:
    embeddings (array, shape (n, d))
    method (str): "pca" (required) or "umap" (optional; only if the
      umap-learn package is installed)

    @return:
    coords (array, shape (n, 2))

    For "pca": use sklearn.decomposition.PCA(n_components=2) and
    fit_transform.  For "umap": import umap and use
    umap.UMAP(n_components=2, random_state=0).fit_transform.
    '''
    raise NotImplementedError("You need to write this part!")


def embedding_rsa(embeddings, reference_sim):
    '''
    Compare the geometry of an embedding space against a reference
    similarity matrix (representational similarity analysis).

    Compute the condensed cosine-distance vector of the embeddings
    (pdist(embeddings, 'cosine')), convert the reference similarity
    matrix to a condensed distance vector
    (squareform(1 - reference_sim, checks=False)), and return the
    Spearman rank correlation between the two.

    @param:
    embeddings (array, shape (n, d))
    reference_sim (array, shape (n, n)): a symmetric similarity matrix
      (1 = identical) from any other source -- another model, a sequence
      comparison, a behavioral measure

    @return:
    r (float): Spearman correlation between the two distance structures
    '''
    raise NotImplementedError("You need to write this part!")


def linear_probe(X, y, groups=None):
    '''
    Measure how much information about y is linearly decodable from a
    frozen representation X, using cross-validated accuracy.

    Use LogisticRegression(max_iter=2000).  If groups is given, use
    GroupKFold(5) with those groups (the honest split when related
    samples are present); otherwise use
    KFold(5, shuffle=True, random_state=0).
    Return the mean accuracy across folds (cross_val_score).

    @param:
    X (array, shape (n, d)): representations (never retrained here)
    y (array, shape (n,)): labels
    groups (array or None): group ids for grouped splitting

    @return:
    acc (float): mean cross-validated accuracy
    '''
    raise NotImplementedError("You need to write this part!")
