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
    raise RuntimeError("You need to write this part!")


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
    raise RuntimeError("You need to write this part!")


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
    raise RuntimeError("You need to write this part!")


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
    raise RuntimeError("You need to write this part!")
