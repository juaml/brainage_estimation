import pickle
import argparse
import pandas as pd
import os.path
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression


def check_predictions(data_df, test_idx, model, test_pred):

    all_idx = np.array(range(0, len(data_df)))
    train_idx = np.delete(all_idx, test_idx)
    train_df = data_df.loc[train_idx, :]
    test_df = data_df.loc[test_idx, :]

    if type(model) == list:
        train_pred = model[0].predict(train_df[X]).ravel()
    else:
        train_pred = model.predict(train_df[X]).ravel()

    print(train_pred.shape, train_df[y].shape)

    # check if test pred saved == test predictions using model
    # print(test_pred)

    test_pred_model = model.predict(test_df[X]).ravel()
    assert(np.round(test_pred) == np.round(test_pred_model)).all()
    print('Predictions match')

def read_data(data_file, demo_file):
    # Read data file: mandatory steps to sort the data (used this to train models)

    demo_df = pd.read_csv(open(demo_file, 'rb'))
    data_df = pickle.load(open(data_file, 'rb'))
    data_df = pd.concat([demo_df, data_df], axis=1)
    data_df = data_df.drop(columns='file_path_cat12.8')

    data_df.rename(columns=lambda X: str(X), inplace=True)  # convert numbers to strings as column names

    X = [col for col in data_df if col.startswith('f_')]
    y = 'age'

    age = data_df['age'].round().astype(int)  # round off age and convert to integer
    data_df['age'] = age
    data_df = data_df[data_df['age'].between(18, 90)].reset_index(drop=True)
    data_df.sort_values(by='age', inplace=True, ignore_index=True)  # or data_df = data_df.reset_index(drop=True)

    print('Any null:', data_df[X].isnull().values.any(), '\n')

    # check for duplicates (multiple sessions for one subject)
    duplicated_subs_1 = data_df[data_df.duplicated(['subject'], keep='first')]
    data_df = data_df.drop(duplicated_subs_1.index).reset_index(drop=True)

    return data_df, X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_nm", type=str, help="Output path for one dataset", default='/ixi/ixi_')

    args = parser.parse_args()
    data_nm = args.data_nm

    # features path, results path and file name for results file
    data_path = '../data'
    results_path = '../results'
    save_file_nm = 'all_models_pred.csv'

    data_list = ['173', '473', '873','1273', 'S0_R4', 'S0_R4', 'S4_R4', 'S4_R4', 'S8_R4', 'S8_R4',
                       'S0_R8', 'S0_R8', 'S4_R8', 'S4_R8', 'S8_R8', 'S8_R8']

    model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly'] #, 'xgb']

    model_names_new = ['RR', 'RFR', 'RVRlin', 'KRR', 'GPR', 'LR', 'ENR', 'RVRpoly'] #, 'XGB'

    filenm_list = ['173', '473', '873','1273', 'S0_R4', 'S0_R4_pca', 'S4_R4', 'S4_R4_pca', 'S8_R4', 'S8_R4_pca',
                       'S0_R8', 'S0_R8_pca', 'S4_R8', 'S4_R8_pca', 'S8_R8', 'S8_R8_pca']

    filenm_list_new = ['173', '473', '873','1273', 'S0_R4', 'S0_R4 + PCA', 'S4_R4', 'S4_R4 + PCA', 'S8_R4', 'S8_R4 + PCA',
                       'S0_R8', 'S0_R8 + PCA', 'S4_R8', 'S4_R8 + PCA', 'S8_R8', 'S8_R8 + PCA']

    # save_file_nm = 'gm_vs_gmwmcsf.csv'
    # data_list = ['S4_R4', 'S4_R4_GM', 'S4_R4_GM_WM_CSF']
    # model_names = ['gauss']
    # model_names_new = ['GPR']
    # filenm_list = ['S4_R4_pca', 'S4_R4_GM_pca', 'S4_R4_GM_WM_CSF_pca'] # our, spm_gm, spm_all
    # filenm_list_new = ['S4_R4 + PCA', 'S4_R4 + PCA + GM', 'S4_R4 + PCA + GM_WM_CSF']


    df_pred_all = pd.DataFrame()
    df_pred = pd.DataFrame()
    df = pd.DataFrame()

    for idx, filenm_item in enumerate(filenm_list):

        for model_item in model_names:
            demo_file = data_path + data_nm + 'subject_list_cat12.8.csv'
            data_file = data_path + data_nm + data_list[idx]
            filenm_result = results_path + data_nm + filenm_item + '_' + model_item + '.results'
            filenm_model = results_path + data_nm + filenm_item + '_' + model_item + '.models'


            print('data file: ', data_file)
            print('demographic file: ', demo_file)
            print('results file: ', filenm_result)
            print('model file:', filenm_model, '\n')


            if os.path.isfile(filenm_model):

                # Read the results file
                res = pickle.load(open(filenm_result,'rb'))
                res_model = pickle.load(open(filenm_model, 'rb'))
                # data_df, X, y = read_data(data_file)
                data_df, X, y = read_data(data_file, demo_file)

                df = pd.DataFrame()
                df_pred = pd.DataFrame()

                for key1, value1 in res.items():
                    df = pd.DataFrame()
                    for key2, value2 in value1.items():
                        print(key1, key2)
                        test_idx = value2['test_idx']
                        print(value2['test_idx'].shape)

                        df['site'] = data_df.iloc[test_idx]['site']
                        df['subject'] = data_df.iloc[test_idx]['subject']
                        df['age'] = data_df.iloc[test_idx]['age']  # should be same as value2['true']
                        # df['true'] = value2['true']
                        df['gender'] = data_df.iloc[test_idx]['gender']

                        if 'session' in data_df.columns:

                            df['session'] = data_df.iloc[test_idx]['session']

                        model = res_model[key1][key2]

                        test_pred = value2['predictions']
                        if (key2 == 'gauss') and (key1 == 'repeat_4'):
                            X_preprocessed, _ = model.preprocess(data_df.loc[test_idx, X], data_df.iloc[test_idx]['age'])
                            print(data_df.loc[test_idx, X].shape, X_preprocessed.shape)

                        check_predictions(data_df, test_idx, model, test_pred)

                        df[filenm_item + ' + ' + key2] = value2['predictions']  # predictions

                    df_pred = pd.concat([df_pred, df], axis=0)

                    df_pred.sort_index(axis=0, level=None, ascending=True, inplace=True)

                if len(df_pred_all) == 0:
                    df_pred_all = df_pred
                else:
                    df_pred_all = df_pred_all.merge(df_pred, on=list(set(data_df.columns.tolist()) - set(X)), how="left")

    save_path = results_path + data_nm + save_file_nm
    print('output path:', save_path)
    df_pred_all.to_csv(save_path, index=False)




