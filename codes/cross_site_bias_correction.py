#!/home/smore/.venvs/py3smore/bin/python3
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from brainage import read_data_cross_site


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == '__main__':
    # Read arguments from submit file
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Data path",
                        default='../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_S4_R4')
    parser.add_argument("--output_filenm", type=str, help="Output file name",
                        default='ixi_camcan_enki_1000brains/4sites_S4_R4_pca_cv.gauss') # path to scores-CV models file
    parser.add_argument("--mod_nm", type=str, help="model name", default='gauss')
    parser.add_argument("--confounds", type=none_or_str, help="confounds", default=None)

    # read arguments
    args = parser.parse_args()
    data = args.data_path
    output_filenm = args.output_filenm
    output_path = '../results/' + output_filenm
    mod_nm = args.mod_nm
    confounds = args.confounds

    # # example arguments
    # data = '../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_S4_R4'
    # output_filenm = 'ixi_camcan_enki_1000brains/4sites_S4_R4_pca_cv.gauss'
    # output_path = '../results/' + output_filenm
    # mod_nm = 'gauss'
    # confounds = None #'site'

    scores_path = output_path + '.scores'
    cv_prediction_savepath = output_path + '_cv_predictions.csv'
    bias_params_savepath = output_path + '_bias_params'

    print('\ninput data:', data)
    print('\noutput_path:', output_path)
    print('\nconfounds:', confounds, type(confounds))
    print('\nscores_path:', scores_path)
    print('\ncv_prediction_savepath:', cv_prediction_savepath)
    print('\nbias_params_savepath:', bias_params_savepath)
    print('\nmodel used:', mod_nm)

    # Load the data which was used for training
    data_df, X, y = read_data_cross_site(data_file=data, train_status='train', confounds=confounds)

    # Fixed variables, set random seed, create classes for age
    rand_seed, n_splits, n_repeats = 200, 5, 5  # fixed during training models
    qc = pd.cut(data_df['age'].tolist(), bins=5, precision=1)  # create bins for train data only
    print('age_bins', qc.categories, 'age_codes', qc.codes)
    data_df['bins'] = qc.codes # add bin/classes as a column in train df

    # Load scores which contains CV models
    scores = pickle.load(open(scores_path, 'rb'))

    # get the exact train and test splits of CV as used during training
    test_idx_all = list()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed).split(data_df, data_df.bins)
    for train_idx, test_idx in cv:
        test_idx_all.append(test_idx)

    # Get CV predictions for each split and repeat
    predictions_df = pd.DataFrame()
    predictions_df_all = pd.DataFrame()

    cv_split = range(0, 25, 5)  # [0, 5, 10, 15, 20] get predictions and arrange them in diff. columns for diff. repeats

    for each_split in cv_split: # for each split (25 in total)
        print('each_split', each_split)

        predictions_df = pd.DataFrame()

        for ind in range(each_split, each_split + n_splits):  # run from (0,5), (5,10), (10,15), (15,20), (20,25)
            print(ind)
            temp_df = pd.DataFrame()
            model_cv = scores[mod_nm]['estimator'][ind] # pick CV estimator
            test_idx = test_idx_all[ind] # pick test indices

            # get predictions for test data
            test_df = data_df.iloc[test_idx, :] # take test data from one split
            y_true = test_df[y]
            y_pred = model_cv.predict(test_df[X]).ravel()
            mae = round(mean_absolute_error(y_true, y_pred), 3)
            mse = round(mean_squared_error(y_true, y_pred), 3)
            corr = round(np.corrcoef(y_pred, y_true)[1, 0], 3)

            print(f' test true age size: {y_true.shape}, predicted age sixe: {y_pred.shape}')
            print(f'MAE: {mae}, MSE: {mse}, CoRR: {corr}')

            if predictions_df.empty:
                predictions_df['test_index'] = pd.Series(test_idx)
                predictions_df['predictions_' + str(each_split)] = pd.Series(y_pred)
            else:
                temp_df['test_index'] = pd.Series(test_idx)
                temp_df['predictions_' + str(each_split)] = pd.Series(y_pred)

            predictions_df = pd.concat([predictions_df, temp_df], axis=0)  # append for all the splits of one repeat

        predictions_df.sort_values(by=['test_index'], inplace=True)
        # print(predictions_df)

        if predictions_df_all.empty:
            predictions_df_all = predictions_df
        else:
            predictions_df_all = predictions_df_all.merge(predictions_df, on=['test_index'], how="left") # merge for all the repeats

    print('predictions_df_all', predictions_df_all)
    predictions_df_all = predictions_df_all.reset_index(drop=True)

    predictions_df_all = pd.concat([data_df[['subject', 'age', 'gender']], predictions_df_all], axis=1) # add subject info
    predictions_df_all.to_csv(cv_prediction_savepath)

    # Calculate m and c from cv predictions for each column
    results_pred = pd.DataFrame()
    filter_col = [col for col in predictions_df_all if col.startswith('predictions')]
    print('filter_col', filter_col)

    model_intercept, model_coef = [], []
    model_bias_params = {'c':0, 'm': 1}

    for column in filter_col:  # for 5 repeats
        X_lin = 'age'
        y_lin = column

        train_x = predictions_df_all.loc[:, X_lin].to_numpy().reshape(-1, 1)  # true age
        train_y = predictions_df_all.loc[:, y_lin].to_numpy().reshape(-1, 1)  # predicted age
        lin_reg = LinearRegression().fit(train_x, train_y)

        # print(lin_reg.intercept_, lin_reg.coef_)
        model_intercept.append(lin_reg.intercept_)
        model_coef.append(lin_reg.coef_)

    # use this m and c for bias correction on test data
    print('average_m', np.mean(model_coef))
    print('average_c', np.mean(model_intercept))

    model_bias_params['m'] = np.mean(model_coef)
    model_bias_params['c'] = np.mean(model_intercept)

    pickle.dump(model_bias_params, open(bias_params_savepath, 'wb'))

    print('ALL DONE')











