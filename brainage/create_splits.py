#!/home/smore/.venvs/py3smore/bin/python3
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


# def create_splits(data_df, repeats):
#     num_bins = math.ceil(len(data_df)/repeats) # calculate number of bins to be created
#     print('num_bins', num_bins, len(data_df)/repeats)
#
#     qc = pd.cut(data_df.index, num_bins)
#     df = pd.DataFrame({'bin': qc.codes})
#
#     max_num = max(df['bin'].value_counts())
#     print(df['bin'].value_counts())
#     print(max_num, 'max_num')
#
#     test_idx = {}
#     for rpt_num in range(0, repeats):
#         key = 'repeat_' + str(rpt_num)
#         test_idx[key] = []
#
#     if repeats == max_num:
#         for num in range(0, max_num):
#             for bin_idx in df['bin'].unique():
#                 test = df[df['bin'] == bin_idx]
#                 if num < len(test):
#                     key = 'repeat_' + str(num)
#                     test_idx[key].append(test.index[num])
#     return test_idx


def stratified_splits(bins_on, num_bins, data, num_splits, shuffle, random_state):
    """
    :param bins_on: variable used to create bins
    :param num_bins: num of bins/classes to create
    :param data: data to create cv splits on
    :param num_splits: number of cv splits to create
    :param shuffle: shuffle the data or not
    :param random_state: random seed to use if shuffle=True
    :return: a dictionary with index
    """
    qc = pd.cut(bins_on.tolist(), num_bins)  # divides data in bins
    cv = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=random_state)
    test_idx = {}
    rpt_num = 0
    for train_index, test_index in cv.split(data, qc.codes):
        key = 'repeat_' + str(rpt_num)
        test_idx[key] = test_index
        rpt_num = rpt_num + 1
    return test_idx


def stratified_splits_class(bins_on, data, num_splits, shuffle, random_state):
    """
    :param bins_on: variable used to create bins
    :param data: data to create cv splits on
    :param num_splits: number of cv splits to create
    :param shuffle: shuffle the data or not
    :param random_state: random seed to use if shuffle=True
    :return: a dictionary with index
    """
    cv = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=random_state)
    test_idx = {}
    rpt_num = 0
    for train_index, test_index in cv.split(data, bins_on):
        key = 'repeat_' + str(rpt_num)
        test_idx[key] = test_index
        rpt_num = rpt_num + 1
    return test_idx


# def stratified_splits(bins_on, num_bins, data, num_splits, shuffle, random_state): # useful for run_cross_validation()
#     """
#     :param bins_on: variable used to create bins
#     :param num_bins: num of bins/classes to create
#     :param data: data to create cv splits on
#     :param num_splits: number of cv splits to create
#     :param shuffle: shuffle the data or not
#     :param random_state: random seed to use if shuffle=True
#     :return: cv iterator
#     """
#     qc = pd.cut(bins_on.tolist(), num_bins)
#     cv = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=random_state).split(data, qc.codes)
#     return cv


def repeated_stratified_splits(bins_on, num_bins, data, num_splits, num_repeats, random_state):
    qc = pd.cut(bins_on.tolist(), num_bins)
    cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=random_state)
    test_idx = {}
    rpt_num = 0
    for train_index, test_index in cv.split(data, qc.codes):
        key = 'repeat_' + str(rpt_num)
        test_idx[key] = test_index
        rpt_num = rpt_num + 1
    return test_idx

