import pickle
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## within site
data_nm = '/ixi/ixi_'
# data_nm = '/enki/enki_'
# data_nm = '/camcan/camcan_'
# data_nm = '/1000brains/1000brains_'

results_folder = '../results'

# Filename to save results
cv_file_ext = 'cv_scores.csv'
test_file_ext = 'test_scores.csv'
combined_file_ext = 'cv_test_scores.csv'

# Complete results filepaths
cv_filename = results_folder + data_nm + cv_file_ext
print('cv_filename', cv_filename)
test_filename = results_folder + data_nm + test_file_ext
print('test_filename', test_filename)
combined_filename = results_folder + data_nm + combined_file_ext
print('combined_filename', combined_filename)


model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly'] #'xgb'
model_names_new = ['RR', 'RFR', 'RVRlin', 'KRR', 'GPR', 'LR', 'ENR', 'RVRpoly'] # 'XGB'

data_list = ['173', '473', '873','1273', 'S0_R4', 'S0_R4_pca', 'S4_R4', 'S4_R4_pca', 'S8_R4', 'S8_R4_pca',
                   'S0_R8', 'S0_R8_pca', 'S4_R8', 'S4_R8_pca', 'S8_R8', 'S8_R8_pca']
data_list_new = ['173', '473', '873','1273', 'S0_R4', 'S0_R4 + PCA', 'S4_R4', 'S4_R4 + PCA', 'S8_R4', 'S8_R4 + PCA',
                   'S0_R8', 'S0_R8 + PCA', 'S4_R8', 'S4_R8 + PCA', 'S8_R8', 'S8_R8 + PCA']

# data_list = ['S4_R4_GM_pca', 'S4_R4_GM_WM_CSF_pca']
# data_list_new = ['S4_R4 + PCA + GM', 'S4_R4 + PCA + GM_WM_CSF']

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
            print(df)
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
            print(df)
            # df_test = df_test.append(df)
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

print(cv_filename)
print(test_filename)
print(combined_filename)

# df_cv.to_csv(cv_filename, index=False)
# df_test.to_csv(test_filename, index=False)
# df_combined.to_csv(combined_filename, index=False)

