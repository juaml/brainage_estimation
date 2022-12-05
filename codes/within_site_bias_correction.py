import math
import os.path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_predictions_file", type=str, help="Path to predictions csv")
    parser.add_argument("--BC_predictions_file", type=str, help="Path to bias corrected predictions")

# python3 within_site_bias_correction.py \
#     --input_predictions_file ../results/ixi/ixi.all_models_pred.csv \
#     --BC_predictions_file ../results/ixi/ixi.all_models_pred_BC.csv
    
    # read arguments
    args = parser.parse_args()
    input_predictions_file = args.input_predictions_file
    BC_predictions_file = args.BC_predictions_file

    # Initialize
    input_df, output_df = pd.DataFrame(), pd.DataFrame()
    column_list, column_name_original = [], []

    if os.path.exists(input_predictions_file):  # if predictions exists
        input_df = pd.read_csv(input_predictions_file) # read predictions from all workflows
        print(input_df.columns)
        print(input_df.index)
        
        if 'session' in input_df.columns:
            column_list = input_df.columns[5:] # remove ['site', 'subject', 'age', 'gender'']
            output_df = input_df[['site', 'subject', 'age', 'gender', 'session']]
        else:
            column_list = input_df.columns[4:] # remove ['site', 'subject', 'age', 'gender'']
            output_df = input_df[['site', 'subject', 'age', 'gender']]

        # Fixed parameters from model training random seed and CV
        rand_seed = 200
        num_splits = 5  # how many train and test splits
        num_bins = math.floor(len(input_df)/num_splits) # num of bins to be created = num of labels created
        qc = pd.cut(input_df.index.tolist(), num_bins) # create bins for age
        cv_5fold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=None)

        for column in column_list:  # for each workflow, X = true age, y= predicted age
            results_pred = pd.DataFrame()
            X = ['age']
            y = column 
            print(f'worflow name: {column}')

            for train_idx, test_idx in cv_5fold.split(input_df, qc.codes):
                # print('test_idx', test_idx)
                train_df, test_df = input_df.loc[train_idx,:], input_df.loc[test_idx,:]  # get test and train dataframes
                print('train size:', train_df.shape, 'test size:', test_df.shape)
                # print(test_df)

                train_x = train_df.loc[:, X]  # true age
                train_y = train_df.loc[:, y]  # predicted age

                model = LinearRegression().fit(train_x, train_y)  # x = age, y = predicted age
                print(model.intercept_, model.coef_)
                corrected_pred = (test_df[y] - model.intercept_) / model.coef_ # corrected predictions

                if results_pred.empty:
                    results_pred = corrected_pred
                else:
                    results_pred = pd.concat([results_pred, corrected_pred], axis=0)

            results_pred.sort_index(axis=0, level=None, ascending=True, inplace=True)
            output_df = pd.concat([output_df, results_pred], axis=1)

        output_df.rename(columns=dict(zip(column_list, column_name_original)), inplace=True)

        print('ALL DONE')
        print(f'Corrected predictions: \n {output_df}')
        output_df.to_csv(BC_predictions_file, index=False)
    else:
        print(f'{input_predictions_file} not found')













