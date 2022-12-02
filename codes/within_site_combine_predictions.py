import pickle
import argparse
import os.path
import numpy as np
import pandas as pd
from brainage import read_data

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

    test_pred_model = model.predict(test_df[X]).ravel()
    assert(np.round(test_pred) == np.round(test_pred_model)).all() # check if test pred saved == test predictions using model

    # print('Prediction from CV models', test_pred)
    # print('Prediction saved during training',test_pred_model)

    print('Predictions match')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demographics_file", type=str, help="Demographics file path")
    parser.add_argument("--features_path", type=str, help="Features file path")
    parser.add_argument("--model_path", type=str, help="Path to directory where within site models of particular datasets are saved")
    parser.add_argument("--output_prefix", type=str, help="Output prefix for predictions filename", default='all_models_pred')
    
    # Parse the arguments
    args = parser.parse_args()
    demographics_file = args.demographics_file
    features_path = args.features_path
    model_path = args.model_path
    output_prefix = args.output_prefix

    # python3 within_site_combine_predictions.py --demographics_file ../data/ixi/ixi.subject_list_cat12.8.csv --features_path ../data/ixi/ixi. --model_path ../results/ixi/ixi. --output_prefix all_models_pred

    # demographics_file = '../data/ixi/ixi.subject_list_cat12.8.csv'
    # features_path = '../data/ixi/ixi.'
    # model_path = '../results/ixi/ixi.'
    # output_prefix = 'all_models_pred'
    
    model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly'] #, 'xgb']
    data_list = ['173', '473', '873','1273', 'S0_R4', 'S0_R4', 'S4_R4', 'S4_R4', 'S8_R4', 'S8_R4',
                       'S0_R8', 'S0_R8', 'S4_R8', 'S4_R8', 'S8_R8', 'S8_R8']
    filenm_list = ['173', '473', '873','1273', 'S0_R4', 'S0_R4_pca', 'S4_R4', 'S4_R4_pca', 'S8_R4', 'S8_R4_pca',
                       'S0_R8', 'S0_R8_pca', 'S4_R8', 'S4_R8_pca', 'S8_R8', 'S8_R8_pca']

    df_pred_all = pd.DataFrame()
    df_pred = pd.DataFrame()
    df = pd.DataFrame()

    for idx, filenm_item in enumerate(filenm_list):  # for each feature space
        for model_item in model_names:  
            features_file = features_path + data_list[idx] # get features file
            result_file = model_path + filenm_item + '.' + model_item + '.results'  # get results
            model_file = model_path + filenm_item + '.' + model_item + '.models'  # get models

            if os.path.isfile(model_file): # if model exists
                print('\n')
                print('data file: ', features_file)
                print('demographic file: ', demographics_file)
                print('model used:', model_file, '\n')
                print('results file: ', result_file)

                # Read the results file
                res = pickle.load(open(result_file,'rb'))  # load the saved results
                res_model = pickle.load(open(model_file, 'rb'))  # load the saved results
                data_df, X, y = read_data(features_file=features_file, demographics_file=demographics_file)

                df = pd.DataFrame()
                df_pred = pd.DataFrame()

                for key1, value1 in res.items():
                    df = pd.DataFrame()
                    for key2, value2 in value1.items():
                        print(key1, key2)
                        test_idx = value2['test_idx'] # get the saved test indices for each fold and pick up demo
                        print(value2['test_idx'].shape)
                        df['site'] = data_df.iloc[test_idx]['site']
                        df['subject'] = data_df.iloc[test_idx]['subject']
                        df['age'] = data_df.iloc[test_idx]['age']  # should be same as value2['true']
                        df['gender'] = data_df.iloc[test_idx]['gender']

                        if 'session' in data_df.columns:
                            df['session'] = data_df.iloc[test_idx]['session']

                        model = res_model[key1][key2]  # get CV model for each fold
                        test_pred = value2['predictions'] # get the saved predictions for each fold

                        check_predictions(data_df, test_idx, model, test_pred) # get predictions using model, check if equal to saved

                        df[filenm_item + ' + ' + key2] = value2['predictions']  # predictions

                    df_pred = pd.concat([df_pred, df], axis=0) # concat over all CV
                    df_pred.sort_index(axis=0, level=None, ascending=True, inplace=True)

                if len(df_pred_all) == 0: # concat over all workflows
                    df_pred_all = df_pred
                else:
                    df_pred_all = df_pred_all.merge(df_pred, on=list(set(data_df.columns.tolist()) - set(X)), how="left")

    print('\n', 'predictions dataframe:', '\n', df_pred_all)
    save_path = model_path + output_prefix + '.csv'
    print('output path:', save_path)
    df_pred_all.to_csv(save_path, index=False)




