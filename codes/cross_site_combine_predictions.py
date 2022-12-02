import pickle
import argparse
import os.path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def model_pred(test_df, X, y, model_file, workflow_name):

    # load the model
    model = pickle.load(open(model_file, 'rb'))
    y_true = test_df[y].reset_index(drop=True)

    # Initialize dataframe for saving output
    pred = pd.DataFrame()
    mae_corr = pd.DataFrame()

    for key, model_value in model.items():
        X_preprocessed, _ = model_value.preprocess(test_df[X], y_true, until='variancethreshold')# until='zscore'
        # print('X_preprocessed shape after variancethreshold',  X_preprocessed.shape)

        # predict test data
        y_pred = model_value.predict(test_df[X]).ravel()
        print('age and predicted age sizes', y_true.shape, y_pred.shape)
        mae = np.round(mean_absolute_error(y_true, y_pred), 3)
        mse = np.round(mean_squared_error(y_true, y_pred), 2)
        corr = np.round(np.corrcoef(y_pred, y_true)[1, 0], 2)

        print('MAE:', mae, 'MSE:', mse, 'CoRR:', corr)
        print('workflow_name:', workflow_name, key)

        pred[workflow_name] = y_pred # add column for predictions
        mae_corr = pd.concat([mae_corr, pd.DataFrame([{'mae': mae, 'mse': mse, 'corr': corr}], index=[workflow_name])], axis=0)

    return pred, y_true, mae_corr


def read_data(features_file, demographics_file):
    demo_df = pd.read_csv(open(demographics_file, 'rb'))
    data_df = pickle.load(open(features_file, 'rb'))
    data_df = pd.concat([demo_df, data_df], axis=1)
    data_df = data_df.drop(columns='file_path_cat12.8')
    data_df.rename(columns=lambda X: str(X), inplace=True)  # convert numbers to strings as column names
    X = [col for col in data_df if col.startswith('f_')]
    y = 'age'
    age = data_df[y].round().astype(int)  # round off age and convert to integer
    data_df[y] = age
    return data_df, X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demographics_file", type=str, help="Demographics file path")
    parser.add_argument("--features_path", type=str, help="Features file path")
    parser.add_argument("--model_path", type=str, help="Path to directory where within site models of particular datasets are saved")
    parser.add_argument("--output_prefix", type=str, help="Output prefix for predictions filename", default='pred_1000brains_all')

    # Parse the arguments
    args = parser.parse_args()
    demographics_file = args.demographics_file
    features_path = args.features_path
    model_path = args.model_path
    output_prefix = args.output_prefix

    # python3 cross_site_combine_predictions.py --demographics_file ../data/1000brains/1000brains.subject_list_cat12.8.csv --features_path ../data/1000brains/1000brains. --model_path ../results/ixi_camcan_enki/ixi_camcan_enki. --output_prefix pred_1000brains_all

    # demographics_file = '../data/1000brains/1000brains.subject_list_cat12.8.csv'
    # features_path = '../data/1000brains/1000brains.'
    # model_path = '../results/ixi_camcan_enki/ixi_camcan_enki.'
    # output_prefix = 'pred_1000brains_all'

    model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly']
    data_list = ['173', '473', '873', '1273', 'S0_R4', 'S0_R4', 'S4_R4', 'S4_R4', 'S8_R4', 'S8_R4',
                 'S0_R8', 'S0_R8', 'S4_R8', 'S4_R8', 'S8_R8', 'S8_R8']
    filenm_list = ['173', '473', '873', '1273', 'S0_R4', 'S0_R4_pca', 'S4_R4', 'S4_R4_pca', 'S8_R4', 'S8_R4_pca',
                 'S0_R8', 'S0_R8_pca', 'S4_R8', 'S4_R8_pca', 'S8_R8', 'S8_R8_pca']

    output_df = pd.DataFrame()
    mae_corr_df = pd.DataFrame()

    for idx, data_item in enumerate(filenm_list): # for each feature space
        for model_item in model_names:
            features_file = features_path + data_list[idx]  # get test features
            model_file = model_path + data_item + '.' + model_item + '.models' # get models

            if os.path.exists(model_file) and os.path.exists(features_file): # if test data and trained model exists
                print('\n')
                print('test data', features_file)
                print('demographic file: ', demographics_file)
                print('model used', model_file)
                print("model and data exists")

                test_df, test_X, test_y = read_data(features_file, demographics_file) # load test data, read data and demo both
                y_pred1, y_true1, mae_corr1 = model_pred(test_df, test_X, test_y, model_file,
                                                         str(data_item + ' + ' + model_item))  # predict test data

                if output_df.empty:
                    needed_cols = test_df.columns[~test_df.columns.isin(test_X)].tolist()
                    output_df = test_df[needed_cols].copy()

                output_df = pd.concat([output_df, y_pred1], axis=1)  # concat for all workflows
                mae_corr_df = pd.concat([mae_corr_df, mae_corr1], axis=0)

    print('\n', 'predictions dataframe:', '\n', output_df)

    mae_corr_df.to_csv(model_path + output_prefix + '_temp.csv')
    output_df.to_csv(model_path + output_prefix + '.csv', index=False)

    # keep predictions from 32 selected workdlows (we trained more than 32)
    selected_workflows_df = ['site', 'subject', 'age', 'gender',
                             '173 + rf', '173 + gauss', '173 + lasso',
                             '473 + lasso', '473 + rvr_poly',
                             '873 + gauss', '873 + elasticnet',
                             '1273 + gauss', '1273 + rvr_poly',
                             'S0_R4 + lasso',
                             'S4_R4 + ridge', 'S4_R4 + rvr_lin', 'S4_R4 + gauss',
                             'S4_R4_pca + ridge', 'S4_R4_pca + rf', 'S4_R4_pca + rvr_lin', 'S4_R4_pca + gauss',
                             'S8_R4 + kernel_ridge',
                             'S8_R4_pca + rvr_lin', 'S8_R4_pca + gauss', 'S8_R4_pca + lasso', 'S8_R4_pca + rvr_poly',
                             'S0_R8 + rvr_poly', 'S0_R8_pca + lasso', 'S0_R8_pca + elasticnet', 'S0_R8_pca + rvr_poly',
                             'S4_R8 + ridge', 'S4_R8 + rvr_lin', 'S4_R8 + lasso',
                             'S8_R8 + ridge', 'S8_R8 + kernel_ridge',
                             'S8_R8_pca + elasticnet']

    if 'session' in output_df.columns:
        selected_workflows_df.insert(4, 'session')

    output_df = output_df.reindex(columns=selected_workflows_df)
    output_df.to_csv( model_path + output_prefix + '_selected' + '.csv', index=False)

