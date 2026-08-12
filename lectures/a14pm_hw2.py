import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_score


def grouped_split(X, y, groups, test_size=0.2):
    '''
    An honest train/test split: related samples never straddle the split.

    Use sklearn's GroupShuffleSplit with n_splits=1, the given test_size,
    and random_state=0 to get one (train_idx, test_idx) pair in which no
    group appears on both sides, then return the four arrays.

    @param:
    X (array, shape (n, d)), y (array, shape (n,))
    groups (array, shape (n,)): group ids (subject, family, batch, ...)
    test_size (float): fraction of samples in the test set

    @return:
    X_train, X_test, y_train, y_test

    Hint: next(GroupShuffleSplit(...).split(X, y, groups)) yields the two
    index arrays.
    '''
    raise RuntimeError("You need to write this part!")


def learning_curve(X, y, sizes):
    '''
    Is performance still climbing with more data, or has it plateaued?

    Shuffle the sample indices once with
    np.random.default_rng(0).permutation(len(y)).  For each n in `sizes`,
    take the first n shuffled samples and compute mean cross-validated
    accuracy with LogisticRegression(max_iter=2000) and
    KFold(5, shuffle=True, random_state=0).  Return the list of
    accuracies, in the order of `sizes`.

    Reading the result: still climbing at the largest n -> collecting more
    of the same data will help; flat -> it won't, and effort belongs
    elsewhere (better features, better labels, a different model).

    @param:
    X (array, shape (n, d)), y (array, shape (n,))
    sizes (list of int): sample sizes to evaluate

    @return:
    accs (list of float), same length as sizes
    '''
    raise RuntimeError("You need to write this part!")


def check_duplicates(X, threshold=0.998):
    '''
    Find near-duplicate pairs of samples in a representation -- the
    quiet destroyers of honest splits.

    Compute all pairwise cosine similarities (1 - cosine distance) and
    return the list of index pairs (i, j) with i < j whose similarity
    exceeds `threshold`.

    Run this on your own data before splitting -- and even when an
    official grouping variable exists, because groupings miss things
    (in this workshop's protein dataset, this function finds paralog
    pairs that sit in different UniRef50 clusters).

    @param:
    X (array, shape (n, d)): one representation per sample
    threshold (float): cosine-similarity cutoff for "near-duplicate"

    @return:
    pairs (list of (int, int) tuples): the offending index pairs

    Hint: squareform(pdist(X, 'cosine')) gives distances;
    np.triu_indices(n, 1) walks each pair once.
    '''
    raise RuntimeError("You need to write this part!")
