# ----------------------------------------------------------------------------
# UNVERIFIED SOLUTION
#
# Written for the workshop -- this is NOT the course author's original.
# It reproduces the expected output published in the notebook, but it has not
# been reviewed by whoever designed the assignment.  Treat it as a reference
# for a TA, not as an answer key.  See solutions/README.md.
# ----------------------------------------------------------------------------
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def fit_polynomial(x, y, degree):
    '''
    Fit a polynomial of the given degree to data, by least squares.

    Build the design matrix H whose columns are [x**0, x**1, ..., x**degree]
    (shape (len(x), degree+1)), then find the coefficient vector minimizing
    the mean-squared error, using np.linalg.lstsq(H, y, rcond=None).

    @param:
    x (array): the input values
    y (array): the observed outputs
    degree (int): the polynomial degree

    @return:
    coefs (array, length degree+1): coefs[k] is the coefficient of x**k

    Hints: np.stack([...], axis=1) builds the design matrix;
    np.linalg.lstsq returns a 4-tuple, and you want its first element.
    '''
    x = np.asarray(x, dtype=float)

    # One column per power: [x^0, x^1, ..., x^degree].  Column k holds the
    # contribution of the k'th coefficient, so H @ coefs is the fitted curve.
    H = np.stack([x ** k for k in range(degree + 1)], axis=1)

    # lstsq finds the coefs minimizing ||H @ coefs - y||^2, which is exactly
    # least squares.  It returns (solution, residuals, rank, singular values);
    # we only want the solution.
    coefs, _residuals, _rank, _sv = np.linalg.lstsq(H, y, rcond=None)
    return coefs


def train_test_error(x, y, degree, split):
    '''
    Estimate training error and test error for a polynomial fit.

    Split the data deterministically: shuffle the indices with
    idx = np.random.default_rng(0).permutation(len(x)), use the first
    round(split*len(x)) shuffled indices as the training set and the rest
    as the test set.  Fit with fit_polynomial on the TRAINING points only.

    @param:
    x, y (arrays): the data
    degree (int): the polynomial degree
    split (float): fraction of points used for training, e.g. 0.7

    @return:
    train_mse (float): mean-squared error on the training points
    test_mse (float): mean-squared error on the test points
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # A fixed seed makes the split reproducible, so everyone in the room gets
    # the same numbers and results can be compared across degrees.
    idx = np.random.default_rng(0).permutation(len(x))
    n_train = round(split * len(x))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Fit on the training points ONLY.  The test points must stay unseen, or
    # the test error stops meaning anything.
    coefs = fit_polynomial(x[train_idx], y[train_idx], degree)

    def mse(sel):
        # Evaluate the polynomial: sum_k coefs[k] * x**k, then average the
        # squared gap between prediction and truth.
        pred = sum(c * x[sel] ** k for k, c in enumerate(coefs))
        return float(np.mean((pred - y[sel]) ** 2))

    return mse(train_idx), mse(test_idx)


def find_best_degree(x, y, degrees):
    '''
    Choose the polynomial degree that generalizes best.

    For each degree in `degrees`, compute the test error using
    train_test_error(x, y, degree, split=0.7), and return the degree
    whose test error is smallest.

    @param:
    x, y (arrays): the data
    degrees (list of int): candidate degrees

    @return:
    best (int): the degree with the smallest test error
    '''
    # Note we compare TEST error, not training error.  Training error keeps
    # dropping as the degree rises -- a high enough polynomial passes through
    # every training point exactly -- so picking on training error would
    # always choose the most complex model.
    test_errors = [train_test_error(x, y, d, split=0.7)[1] for d in degrees]
    return degrees[int(np.argmin(test_errors))]


def honest_vs_leaky(X, y):
    '''
    Measure how much accuracy is inflated when feature selection happens
    before the train/test split instead of after.

    Both pipelines select the 20 features whose Pearson correlation with y
    has the largest absolute value, split with
    train_test_split(..., test_size=0.3, random_state=0), fit
    LogisticRegression(max_iter=1000) on the training rows, and score on
    the test rows.  The ONLY difference is where selection happens:

    leaky:  select the 20 features using ALL rows, then split X[:, selected].
    honest: split X first, select the 20 features using the TRAINING rows
            only, then apply that same selection to the test rows.

    @param:
    X (array, shape (n, d)): feature matrix
    y (array, shape (n,)): binary labels (0/1)

    @return:
    honest_acc (float): test accuracy of the honest pipeline
    leaky_acc (float): test accuracy of the leaky pipeline

    Hint: np.corrcoef(X[:, j], y)[0, 1] is the correlation of feature j
    with the labels; np.argsort(np.abs(r))[-20:] gives the top 20.
    '''
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    def top20(rows_X, rows_y):
        # Correlation of every feature with the labels, then the 20 whose
        # correlation is largest in absolute value (sign does not matter --
        # a strongly negative correlation is just as informative).
        r = np.array([np.corrcoef(rows_X[:, j], rows_y)[0, 1]
                      for j in range(rows_X.shape[1])])
        r = np.nan_to_num(r)          # a constant feature gives NaN
        return np.argsort(np.abs(r))[-20:]

    def accuracy(Xtr, Xte, ytr, yte):
        model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        return float(model.score(Xte, yte))

    # LEAKY: the selection looks at every row, including the ones we are about
    # to call "test".  Nothing is copied across the split, but the CHOICE of
    # features was informed by the test labels, and that is enough to inflate
    # the score.
    sel_all = top20(X, y)
    Xtr, Xte, ytr, yte = train_test_split(X[:, sel_all], y,
                                          test_size=0.3, random_state=0)
    leaky_acc = accuracy(Xtr, Xte, ytr, yte)

    # HONEST: split first, and let the training rows alone decide which
    # features to keep.  The same columns are then read off the test rows.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    sel_train = top20(Xtr, ytr)
    honest_acc = accuracy(Xtr[:, sel_train], Xte[:, sel_train], ytr, yte)

    return honest_acc, leaky_acc
