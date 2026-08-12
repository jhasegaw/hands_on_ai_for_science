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
    raise RuntimeError("You need to write this part!")


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
    raise RuntimeError("You need to write this part!")


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
    raise RuntimeError("You need to write this part!")
