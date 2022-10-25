#!/home/smore/.venvs/py3smore/bin/python3
import math
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, LeaveOneOut, StratifiedKFold
from julearn import run_cross_validation
# import pingouin as pg
import numpy as np
from sklearn.linear_model import LinearRegression
import os.path
import scipy.stats

# Variable parameter
# dataset_flag = 'enki' # 'enki

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_flag", type=str, help="Output path for one dataset", default='ixi')

    args = parser.parse_args()
    dataset_flag = args.dataset_flag

    # data and results path
    results_path = '../results/'

    # Initialize
    data_df, results_all = pd.DataFrame(), pd.DataFrame()
    column_list, column_name_original = [], []
    output_path, output_path_corr = '', ''
    data = ''

    if dataset_flag == 'ixi':
        data = results_path + 'ixi/ixi_all_models_pred.csv'
        output_path = results_path + 'ixi/ixi_all_models_pred_BC.csv'

    elif dataset_flag == 'enki':
        data = results_path + 'enki/enki_all_models_pred.csv'
        output_path = results_path + 'enki/enki_all_models_pred_BC.csv'

    elif dataset_flag == 'camcan':
        data = results_path + 'camcan/camcan_all_models_pred.csv'
        output_path = results_path + 'camcan/camcan_all_models_pred_BC.csv'

    elif dataset_flag == '1000brains':
        data = results_path + '1000brains/1000brains_all_models_pred.csv'
        output_path = results_path + '1000brains/1000brains_all_models_pred_BC.csv'
    else:
        print('error')

    if os.path.exists(data):
        data_df = pd.read_csv(data, ',') # read predictions
        print(data_df.columns)
        print(data_df.index)

        # column_name_original = data_df.columns[4:] # get original names of workflows
        # data_df.columns = data_df.columns.str.replace(r"+", "_")  # change + to _ in workflow names

        if 'session' in data_df.columns:
            column_list = data_df.columns[5:] # remove ['site', 'subject', 'age', 'gender'']
            results_all = data_df[['site', 'subject', 'age', 'gender', 'session']]
        else:
            column_list = data_df.columns[4:] # remove ['site', 'subject', 'age', 'gender'']
            results_all = data_df[['site', 'subject', 'age', 'gender']]

        # initialize random seed and CV
        rand_seed = 200
        num_splits = 5  # how many train and test splits
        num_bins = math.floor(len(data_df)/num_splits) # num of bins to be created = num of labels created
        qc = pd.cut(data_df.index.tolist(), num_bins)
        cv_5fold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=None)

        for column in column_list:
            results_pred = pd.DataFrame()
            X = ['age']
            y = column
            print(column)
            cv_5fold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=None)

            for train_idx, test_idx in cv_5fold.split(data_df, qc.codes):
                # print('test_idx', test_idx)
                train_df, test_df = data_df.loc[train_idx,:], data_df.loc[test_idx,:]  # get test and train dataframes
                print('train size:', train_df.shape, 'test size:', test_df.shape)
                # print(test_df)

                train_x = train_df.loc[:, X]
                train_y = train_df.loc[:, y]

                model = LinearRegression().fit(train_x, train_y)
                print(model.intercept_, model.coef_)

                corrected_pred = (test_df[y] - model.intercept_) / model.coef_

                if results_pred.empty:
                    results_pred = corrected_pred
                else:
                    results_pred = pd.concat([results_pred, corrected_pred], axis=0)

            results_pred.sort_index(axis=0, level=None, ascending=True, inplace=True)
            results_all = pd.concat([results_all, results_pred], axis=1)

        results_all.rename(columns=dict(zip(column_list, column_name_original)), inplace=True)

        print('ALL DONE')
        results_all.to_csv(output_path, index=False)
    else:
        print(f'{data} not found')













