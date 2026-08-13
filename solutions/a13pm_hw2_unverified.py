# ----------------------------------------------------------------------------
# UNVERIFIED SOLUTION
#
# Written for the workshop -- this is NOT the course author's original.
# It reproduces the expected output published in the notebook, but it has not
# been reviewed by whoever designed the assignment.  Treat it as a reference
# for a TA, not as an answer key.  See solutions/README.md.
# ----------------------------------------------------------------------------
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
    # pdist returns distances in "condensed" form (one long vector of every
    # pair); squareform expands it into the (n, n) matrix that is easier to
    # index.  Cosine distance ignores vector length and compares direction
    # only, which is what you want for embeddings.
    dists = squareform(pdist(embeddings, 'cosine'))

    # Row `query` holds this item's distance to everything.  Sorting it gives
    # nearest first -- but position 0 is always the query compared with
    # itself (distance 0), so drop that and keep the next k.
    order = np.argsort(dists[query])
    order = order[order != query]
    return order[:k]


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
    if method == "pca":
        # PCA keeps the two directions along which the embeddings vary most.
        # It is linear, so distances in the plot understate how separated the
        # groups really are in the full space -- useful for looking, not for
        # measuring.
        return PCA(n_components=2).fit_transform(embeddings)

    if method == "umap":
        # Optional: UMAP preserves local neighbourhoods better, at the cost of
        # distances between distant clusters becoming meaningless.
        import umap
        return umap.UMAP(n_components=2, random_state=0).fit_transform(embeddings)

    raise ValueError("method must be 'pca' or 'umap', got %r" % (method,))


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
    # Both sides must be in the SAME condensed form, so that element i of one
    # vector describes the same pair of items as element i of the other.
    emb_d = pdist(embeddings, 'cosine')

    # The reference arrives as similarity (1 = identical); 1 - sim turns it
    # into distance.  checks=False skips squareform's symmetry validation,
    # which floating-point noise can otherwise trip.
    ref_d = squareform(1 - np.asarray(reference_sim), checks=False)

    # Spearman compares RANKS, so it asks the question that matters here: do
    # the two spaces agree about which pairs are closer than which, regardless
    # of the units either one uses?
    r, _p = spearmanr(emb_d, ref_d)
    return float(r)


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
    model = LogisticRegression(max_iter=2000)

    # With groups, every member of a group lands in the same fold, so a model
    # can never be tested on something closely related to what it trained on.
    # Without groups, a plain shuffled 5-fold is fine.
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)

    # The representation X is never retrained -- only the little logistic
    # regression on top is.  That is what makes this a probe: it measures what
    # is already in the representation, not what a model could learn.
    scores = cross_val_score(model, X, y, groups=groups, cv=cv)
    return float(scores.mean())
