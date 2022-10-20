import pickle
import pandas as pd
import os.path
import numpy as np
from scipy.stats import zscore
import math
import argparse
import os.path
from sklearn.metrics import mean_absolute_error, mean_squared_error


def model_pred(test_df, X, y, model_file, file_name):
    # load the model
    model = pickle.load(open(model_file, 'rb'))
    y_true = test_df[y].reset_index(drop=True)

    # Initialize dataframe for saving output
    pred = pd.DataFrame()
    mae_corr = pd.DataFrame()

    for key, model_value in model.items():
        X_preprocessed, _ = model_value.preprocess(test_df[X], y_true, until='variancethreshold')# until='zscore'
        # X_preprocessed2, _ = model_value.preprocess(test_df[X], y_true, until='pca')  # until='zscore'
        print('X_preprocessed shape after variancethreshold',  X_preprocessed.shape)
        # print('X_preprocessed shape after variancethreshold ans zscore',  X_preprocessed2.shape)

        # predict test data
        y_pred = model_value.predict(test_df[X]).ravel()

        print('age and predicted age sizes', y_true.shape, y_pred.shape)
        mae = np.round(mean_absolute_error(y_true, y_pred), 3)
        mse = np.round(mean_squared_error(y_true, y_pred), 2)
        corr = np.round(np.corrcoef(y_pred, y_true)[1, 0], 2)
        print('----------', mae, mse, corr)
        print(file_name, key)

        pred[file_name] = y_pred # add column for predictions
        mae_corr = pd.concat([mae_corr, pd.DataFrame([{'mae': mae, 'mse': mse, 'corr': corr}], index=[file_name])], axis=0)

    return pred, y_true, mae_corr


def data_read(*args): #data, confounds, ses
    # data, confounds = data_file, confounds
    data = args[0]
    confounds = args[1]

    data_df = pickle.load(open(data, 'rb'))
    data_df.rename(columns=lambda X: str(X), inplace=True)  # convert numbers to strings as column names

    age = data_df['age'].round().astype(int)  # round off age and convert to integer
    data_df['age'] = age
    data_df = data_df[data_df['age'].between(18, 90)].reset_index(drop=True)

    X = [col for col in data_df if col.startswith('f_')]
    y = 'age'
    print(data_df.shape)
    return data_df, X, y


if __name__ == '__main__':

    confounds = None
    # train_data_name = '/5_datasets/5_datasets_' # model trained with 5 datasets
    # data_path = '/data/project/brainage/data_new/oasis3/old/oasis3_comp_brainageR_' # features with few subs to match oasis3_pet sub list

    data_path = '/data/project/brainage/brainage_julearn_final/data_new'
    results_folder = '/data/project/brainage/brainage_julearn_final/results'

    # train_data_name = '/ixi_camcan_enki/ixi_camcan_enki_' # model trained with 4 datasets
    # model_folder = '/ixi_camcan_enki/ixi_camcan_enki_'
    # test_data_name = '/1000brains/new/1000brains_'
    # save_file_ext = 'pred_1000brains_all'

    # train_data_name = '/camcan_enki_1000brains/camcan_enki_1000brains_' # model trained with 4 datasets
    # model_folder = '/camcan_enki_1000brains/camcan_enki_1000brains_'
    # test_data_name = '/ixi/new/ixi_'
    # save_file_ext = 'pred_ixi_all'

    train_data_name = '/ixi_camcan_1000brains/ixi_camcan_1000brains_' # model trained with 4 datasets
    model_folder = '/ixi_camcan_1000brains/ixi_camcan_1000brains_'
    test_data_name = '/enki/new/enki_'
    save_file_ext = 'pred_enki_all'

    # train_data_name = '/ixi_enki_1000brains/ixi_enki_1000brains_' # model trained with 4 datasets
    # model_folder = '/ixi_enki_1000brains/ixi_enki_1000brains_'
    # test_data_name = '/camcan/new/camcan_'
    # save_file_ext = 'pred_camcan_all'

    # train_data_name = '/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_' # model trained with 4 datasets
    # model_folder = '/ixi_camcan_enki_1000brains/4sites_'
    # test_data_name = '/oasis3/new/oasis3_'
    # save_file_ext = 'pred_oasis3_all'

    # train_data_name = '/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_' # model trained with 4 datasets
    # model_folder = '/ixi_camcan_enki_1000brains/4sites_'
    # test_data_name = '/corr/new/corr_'
    # save_file_ext = 'pred_corr_all'

    # train_data_name = '/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_' # model trained with 4 datasets
    # model_folder = '/ixi_camcan_enki_1000brains/4sites_'
    # test_data_name = '/adni/new/ADNI_'
    # save_file_ext = 'pred_adni_all'

    test_data_path = data_path + test_data_name
    train_data_path = data_path + train_data_name

    model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly']
    data_list = ['173', '473', '873', '1273', 'S0_R4', 'S0_R4_pca', 'S4_R4', 'S4_R4_pca', 'S8_R4', 'S8_R4_pca',
                 'S0_R8', 'S0_R8_pca', 'S4_R8', 'S4_R8_pca', 'S8_R8', 'S8_R8_pca']
    data_list_data = ['173', '473', '873', '1273', 'S0_R4', 'S0_R4', 'S4_R4', 'S4_R4', 'S8_R4', 'S8_R4',
                 'S0_R8', 'S0_R8', 'S4_R8', 'S4_R8', 'S8_R8', 'S8_R8']


    output_df = pd.DataFrame()
    mae_corr_df = pd.DataFrame()

    for idx, data_item in enumerate(data_list):
        for model_item in model_names:
            test_data = test_data_path + data_list_data[idx]
            train_data = train_data_path + data_list_data[idx] # not needed

            model_file = results_folder + model_folder + data_item + '.' + model_item + '.models'

            if os.path.exists(model_file) and os.path.exists(test_data):
                print(train_data)
                print(test_data)
                print(model_file)
                print("model and data exists")

                # train_df, train_X, train_y = data_read(train_data, confounds)  # load train data
                test_df, test_X, test_y = data_read(test_data, confounds) # load test data
                y_pred1, y_true1, mae_corr1 = model_pred(test_df, test_X, test_y, model_file,
                                                         str(data_item + ' + ' + model_item)) # predict test data

                if output_df.empty:
                    # output_df = test_df[test_df.columns[0:4]].reset_index(drop=True)
                    needed_cols = test_df.columns[~test_df.columns.isin(test_X)].tolist()
                    output_df = test_df[needed_cols].copy()

                output_df = pd.concat([output_df, y_pred1], axis=1)
                mae_corr_df = pd.concat([mae_corr_df, mae_corr1], axis=0)

    mae_corr_df.to_csv(results_folder + model_folder+ save_file_ext + '_temp.csv')
    output_df.to_csv(results_folder + model_folder + save_file_ext + '.csv', index=False)

    if 'session' is output_df.columns:
        selected_workflows_df = ['site', 'subject', 'age', 'gender', 'session',
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
    else:
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

    output_df = output_df.reindex(columns=selected_workflows_df)
    output_df.to_csv(results_folder + model_folder + save_file_ext + '_selected' + '.csv', index=False)

