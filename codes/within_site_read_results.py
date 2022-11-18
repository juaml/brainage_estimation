import pickle
import pandas as pd
import os.path
import numpy as np
import argparse

# Input options possible
## within site
# data_nm = '/ixi/ixi_'
# data_nm = '/enki/enki_'
# data_nm = '/camcan/camcan_'
# data_nm = '/1000brains/1000brains_'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_nm", type=str, help="Output path for one dataset", default='/ixi/ixi_')

    args = parser.parse_args()
    data_nm = args.data_nm

    results_folder = '../results'

    # Filename to save results
    cv_file_ext = 'cv_scores.csv'
    test_file_ext = 'test_scores.csv'
    combined_file_ext = 'cv_test_scores.csv'

    # Complete results filepaths
    cv_filename = results_folder + data_nm + cv_file_ext
    test_filename = results_folder + data_nm + test_file_ext
    combined_filename = results_folder + data_nm + combined_file_ext

    # all model names
    model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly'] #'xgb'
    model_names_new = ['RR', 'RFR', 'RVRlin', 'KRR', 'GPR', 'LR', 'ENR', 'RVRpoly'] # 'XGB'

    # all feature spaces names
    data_list = ['173', '473', '873','1273', 'S0_R4', 'S0_R4_pca', 'S4_R4', 'S4_R4_pca', 'S8_R4', 'S8_R4_pca',
                       'S0_R8', 'S0_R8_pca', 'S4_R8', 'S4_R8_pca', 'S8_R8', 'S8_R8_pca']
    data_list_new = ['173', '473', '873','1273', 'S0_R4', 'S0_R4 + PCA', 'S4_R4', 'S4_R4 + PCA', 'S8_R4', 'S8_R4 + PCA',
                       'S0_R8', 'S0_R8 + PCA', 'S4_R8', 'S4_R8 + PCA', 'S8_R8', 'S8_R8 + PCA']

    # check which scores file is missing
    missing_outs = []
    for data_item in data_list:
        for model_item in model_names:
            scores_item = results_folder + data_nm + data_item + '_' + model_item + '.scores' # create the complete path to scores file
            if os.path.isfile(scores_item):
                print('yes')
            else:
                missing_outs.append(scores_item)
    print('Missing files:\n', missing_outs)


    # get the saved cv scores
    df = pd.DataFrame()
    df_cv = pd.DataFrame()
    for data_item in data_list:
        for model_item in model_names:
            scores_item = results_folder + data_nm + data_item + '_' + model_item + '.scores' # create the complete path to scores file
            if os.path.isfile(scores_item):
                print(scores_item)
                res = pickle.load(open(scores_item,'rb'))
                df = pd.DataFrame()
                for key1, value1 in res.items():
                    print('key1', key1)
                    mae_all, mse_all, corr_all, corr_delta_all, key_all = list(), list(), list(), list(), list()
                    for key, value in value1.items():
                        mae = round(value['test_neg_mean_absolute_error'].mean() * -1, 3)
                        mse = round(value['test_neg_mean_squared_error'].mean() * -1, 3)
                        corr = round(value['test_r2'].mean(), 3)
                        mae_all.append(mae)
                        mse_all.append(mse)
                        corr_all.append(corr)
                        key_all.append(key)

                    df['model'] = key_all
                    df['data'] = len(mae_all) * [data_item]
                    df[key1 + '_mae'] = mae_all
                    df[key1 + '_mse'] = mse_all
                    df[key1 + '_corr'] = corr_all
                # print(df)
                df_cv = pd.concat([df_cv, df], axis=0)

    df_cv.reset_index(drop=True, inplace=True)

    xx_mae = df_cv.loc[:, df_cv.columns.str.endswith('_mae')].values # ro take average over repeats of mae
    xx_mse = df_cv.loc[:, df_cv.columns.str.endswith('_mse')].values # ro take average over repeats of mae
    xx_corr = df_cv.loc[:, df_cv.columns.str.endswith('_corr')].values # ro take average over repeats of mae

    df_cv['mean_cv_mae'] = np.mean(xx_mae, axis=1).round(3)
    df_cv['mean_cv_mse'] = np.mean(xx_mse, axis=1).round(3)
    df_cv['mean_cv_corr'] = np.mean(xx_corr, axis=1).round(3)

    df_cv['workflow_name'] = df_cv['data'] + ' + ' + df_cv['model']
    df_cv['data'] = df_cv['data'].replace(data_list, data_list_new)
    df_cv['model'] = df_cv['model'].replace(model_names, model_names_new)


    # # get the saved test scores
    df = pd.DataFrame()
    df_test = pd.DataFrame()

    for data_item in data_list:
        for model_item in model_names:
            scores_item = results_folder + data_nm + data_item + '_' + model_item + '.results' # create the complete path to scores file
            if os.path.isfile(scores_item):
                print(scores_item)
                res = pickle.load(open(scores_item,'rb'))
                df = pd.DataFrame()
                for key1, value1 in res.items():
                    print('key1', key1)
                    mae_all, mse_all, corr_all, key_all = list(), list(), list(), list()
                    for key, value in value1.items():
                        mae = value['mae']
                        mse = value['mse']
                        corr = value['corr']
                        mae_all.append(mae)
                        mse_all.append(mse)
                        corr_all.append(corr)
                        key_all.append(key)
                    df['model'] = key_all
                    df['data'] = len(mae_all) * [data_item]
                    df[key1 + '_mae'] = mae_all
                    df[key1 + '_mse'] = mse_all
                    df[key1 + '_corr'] = corr_all
                # print(df)
                df_test = pd.concat([df_test, df], axis=0)

    df_test.reset_index(drop=True, inplace=True)

    xx_mae = df_test.loc[:, df_test.columns.str.endswith('_mae')].values # ro take average over repeats of mae
    xx_mse = df_test.loc[:, df_test.columns.str.endswith('_mse')].values # ro take average over repeats of mae
    xx_corr = df_test.loc[:, df_test.columns.str.endswith('_corr')].values # ro take average over repeats of mae

    df_test['mean_test_mae'] = np.mean(xx_mae, axis=1).round(3)
    df_test['mean_test_mse'] = np.mean(xx_mse, axis=1).round(3)
    df_test['mean_test_corr'] = np.mean(xx_corr, axis=1).round(3)

    df_test['workflow_name'] = df_test['data'] + ' + ' + df_test['model']
    df_test['data'] = df_test['data'].replace(data_list, data_list_new)
    df_test['model'] = df_test['model'].replace(model_names, model_names_new)

    df_combined1 = df_cv[['model', 'data', 'mean_cv_mae', 'mean_cv_mse', 'mean_cv_corr', 'workflow_name']].copy()
    df_combined2 = df_test[['model', 'data', 'mean_test_mae', 'mean_test_mse', 'mean_test_corr', 'workflow_name']].copy()
    df_combined = pd.merge(df_combined1, df_combined2, how='left', on=['model', 'data', 'workflow_name'])
    df_combined['workflow_name_updated'] = df_combined['data'] + ' + ' + df_combined['model']
    df_combined.reset_index(drop=True, inplace=True)

    # save the csv files
    print('\n cv results file:', cv_filename)
    print(df_cv)
    print('\n test results file:', test_filename)
    print(df_test)
    print('\n combined results file:', combined_filename)
    print(df_combined)

    df_cv.to_csv(cv_filename, index=False)
    df_test.to_csv(test_filename, index=False)
    df_combined.to_csv(combined_filename, index=False)


    # # check model parameters
    print('\n Model Parameters')
    error_models = list()
    for data_item in data_list:
        for model_item in model_names:
            model_item = results_folder + data_nm + data_item + '_' + model_item + '.models'  # get models

            if os.path.isfile(model_item):
                print('\n', 'model filename', model_item)

                res = pickle.load(open(model_item, 'rb'))
                # print(res)

                for key1, value1 in res.items():
                    for key2, value2 in value1.items():
                        print(key1, key2)

                        if key2 == 'linreg':
                            print(res[key1]['linreg']['linreg'].intercept_, res[key1]['linreg']['linreg'].coef_)

                        elif key2 == 'gauss':
                            model = res[key1]['gauss']['gauss']
                            # print(model.get_params())
                            print(model.kernel_.get_params())

                        elif key2 == 'kernel_ridge':
                            model = res[key1]['kernel_ridge']['kernelridge']
                            print(model)

                        elif key2 == 'rvr_lin':
                            model = res[key1]['rvr_lin']['rvr']
                            print(model)

                        elif key2 == 'rvr_poly':
                            model = res[key1]['rvr_poly']['rvr']
                            print(model)

                        elif key2 == 'rf':
                            model = res[key1]['rf']['rf']
                            print(model)

                        elif key2 == 'xgb':
                            model = res[key1]['xgb']['xgboostadapted']
                            print(model)

                        else:  # for lasso, ridge, elasticnet
                            model = res[key1][key2]['elasticnet']
                            print(model.lambda_best_)

            else:
                error_models.append(model_item)



