import numpy as np
import pandas as pd


def load_and_clean(path):
    '''
    Load a messy measurements CSV and clean it up.

    @param:
    path (str): path to a CSV file formatted like a12pm_measurements.csv

    @return:
    df (DataFrame): the loaded table, with two problems fixed:
      1. The `condition` column is lowercased and stripped of surrounding
         whitespace, so that it contains exactly two values:
         'control' and 'treatment'.
      2. The `gene_d` column is converted to numeric.  Entries that are not
         numbers (like 'n.d.') become NaN.
    Do not drop any rows: the returned DataFrame has the same number of rows
    as the file.

    Hints: .str.lower(), .str.strip(), pd.to_numeric(..., errors='coerce')
    '''
    df = pd.read_csv(path)

    # Problem 1: the condition column holds eight spellings of two words --
    # 'control', 'Control', 'CONTROL', and 'control ' with a trailing space.
    # Lowercasing collapses the capitalisation and .str.strip() removes the
    # invisible surrounding whitespace, leaving exactly two labels.
    df['condition'] = df['condition'].str.lower().str.strip()

    # Problem 2: gene_d arrived as text, because a few cells say 'n.d.'
    # ("not detected") and one non-number is enough to make pandas store the
    # whole column as strings.  errors='coerce' means "if you cannot read it
    # as a number, put NaN there" -- which is honest, since those values
    # really are missing.
    df['gene_d'] = pd.to_numeric(df['gene_d'], errors='coerce')

    # Note what we did NOT do: no rows were removed.  Deciding which rows to
    # discard is the caller's job, and different analyses want different
    # things.  Cleaning and filtering are separate steps.
    return df


def to_feature_matrix(df, feature_cols, label_col):
    '''
    Convert a cleaned DataFrame into the (X, y) arrays that every model in
    this workshop eats.

    @param:
    df (DataFrame): a cleaned table, e.g. the output of load_and_clean
    feature_cols (list of str): names of the numeric feature columns
    label_col (str): name of the label column

    @return:
    X (ndarray of float, shape (n, len(feature_cols))): feature matrix
    y (ndarray, shape (n,)): label values (strings are fine)

    Rows that have NaN in ANY of the feature columns are dropped from both
    X and y (so the two arrays stay aligned).

    Hints: df.dropna(subset=...), .to_numpy()
    '''
    # Drop incomplete rows FIRST, while features and labels are still side by
    # side in one table.  That is what keeps X and y aligned: row i of X and
    # element i of y come from the same surviving row.  Splitting first and
    # dropping afterwards is the classic way to silently mismatch them.
    #
    # subset=feature_cols means "only look at the feature columns when
    # deciding what is incomplete" -- a missing label is a different problem.
    complete = df.dropna(subset=feature_cols)

    # .to_numpy() hands back a plain numpy array, which is what the models
    # want.  The features become a 2D array with one row per sample and one
    # column per gene; the labels become a 1D array of strings.
    X = complete[feature_cols].to_numpy(dtype=float)
    y = complete[label_col].to_numpy()

    return X, y


def group_means(df, by, value):
    '''
    Compute the mean of one column, separately for each level of another.

    @param:
    df (DataFrame): the data table
    by (str): name of the categorical column to group by
    value (str): name of the numeric column to average

    @return:
    means (Series): the mean of `value` for each level of `by`, indexed by
      the levels of `by` -- i.e., the result of df.groupby(by)[value].mean()
    '''
    # groupby splits the table into one group per distinct value of `by`,
    # picking the `value` column out of each, and .mean() collapses each group
    # to a single number.  The result is indexed by the group labels, so you
    # can read it as a small table.
    #
    # pandas skips NaN when averaging, so missing measurements simply do not
    # contribute -- there is no need to filter them out first.  This is why
    # you run this on the full cleaned table rather than on the smaller set of
    # rows that survived to_feature_matrix.
    return df.groupby(by)[value].mean()
