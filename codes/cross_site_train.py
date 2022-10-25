import pickle
import math
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.transformers import register_transformer
from sklearn.feature_selection import VarianceThreshold

from skrvm import RVR
import sklearn.gaussian_process as gp
from sklearn.kernel_ridge import KernelRidge
from glmnet import ElasticNet
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
#from xgboost_adapted import XGBoostAdapted
from brainage import XGBoostAdapted
from pathlib import Path
import os.path
import time

start_time = time.time()


def read_data(data_file, train_status):
    data_df = pickle.load(open(data_file, 'rb'))

    X = [col for col in data_df if col.startswith('f_')]
    y = 'age'
    data_df['age'] = data_df['age'].round().astype(int)  # round off age and convert to integer
    data_df = data_df[data_df['age'].between(18, 90)].reset_index(drop=True)
    duplicated_subs_1 = data_df[data_df.duplicated(['subject'], keep='first')] # check for duplicates (multiple sessions for one subject)
    data_df = data_df.drop(duplicated_subs_1.index).reset_index(drop=True)  # remove duplicated subjects

    if confounds is not None:  # convert sites in numbers to perform confound removal

        if train_status == 'train':
            site_name = data_df['site'].unique()
            if type(site_name[0]) == str:
                site_dict = {k: idx for idx, k in enumerate(site_name)}
                data_df['site'] = data_df['site'].replace(site_dict)

        elif train_status == 'test': # add site to features & convert site in a number to predict with model trained with  confound removal
            X.append(confounds)
            site_name = data_df['site'].unique()[0,]
            if type(site_name) == str:
                data_df['site'] = 10
    return data_df, X, y


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == '__main__':

    # Read arguments from submit file
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Data path")
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--models", type=str, nargs='?', const=1, default="RR",
                       help="models to use (comma seperated no space): RR,LinearSVC")
    # parser.add_argument("--test_data_path", type=str, help="Test Data path")
    parser.add_argument("--confounds", type=none_or_str, help="confounds", default=None)
    parser.add_argument("--pca_status", type=int, default=0,
                       help="0: no pca, 1: yes pca")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run")

    configure_logging(level='INFO')

    args = parser.parse_args()
    data = args.data_path
    # test_data = args.test_data_path
    output_path = args.output_path
    model_required = [x.strip() for x in args.models.split(',')]  # converts string into list
    confounds = args.confounds
    pca_status = bool(args.pca_status)
    n_jobs = args.n_jobs

    # data = '../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_173'
    # test_data = '../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_173'
    # output_path = '../results/ixi/ixi_test'
    # model_required = ['rvr_lin']
    # confounds = None
    # pca_status = bool(0)
    # n_jobs = 1
    
    output_dir, output_file = os.path.split(output_path)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # initialize random seed and create test indices
    rand_seed = 200
    n_repeats = 5 # for inner CV
    n_splits = 5  # how many train and test splits (both for other and inner)

    print('Data file:', data)
    print ('Output path : ', output_dir)
    # print('test_data:', test_data)
    print('Model:', model_required, type(model_required))
    print ('PCA status : ', pca_status)
    print ('Random seed : ', rand_seed)
    print ('Num of splits for kfolds : ', n_splits, '\n')
    print('confounds:', confounds, type(confounds))
    print('Num of parallel jobs initiated: ', n_jobs, '\n')
    
    configure_logging(level='INFO')

    # Load the train data
    data_df, X, y = read_data(data_file=data, train_status='train')

    # take only samples with age 60 to 80 years
    # data_df = data_df[(data_df['age'] >= 60) & (data_df['age'] <= 90)]

    # Load the test data
    # test_df, X_test, y_test = read_data(data_file=test_data, train_status='test')
    # output_df = test_df[['site', 'subject', 'age', 'gender']]

    # Initialize variables, set random seed, create classes for age
    scores_cv, models, results = {}, {}, {}
    qc = pd.cut(data_df['age'].tolist(), bins=5, precision=1)  # create bins for train data only
    print('age_bins', qc.categories, 'age_codes', qc.codes)
    data_df['bins'] = qc.codes # add bin/classes as a column in train df

    # register VarianceThreshold as a transformer
    register_transformer('variancethreshold', VarianceThreshold, returned_features='unknown',
                         apply_to='all_features')
    var_threshold = 1e-5

    # Define all models
    rvr_linear = RVR()
    rvr_poly = RVR()
    kernel_ridge = KernelRidge()  # kernelridge
    lasso = ElasticNet(alpha=1, standardize=False)
    elasticnet = ElasticNet(alpha=0.5, standardize=False)
    ridge = ElasticNet(alpha=0, standardize=False)
    xgb = XGBoostAdapted(early_stopping_rounds=10, eval_metric='mae', eval_set_percent=0.2)
    pca = PCA(n_components=None)  # max as many components as sample size

    model_list = [ridge, 'rf', rvr_linear, kernel_ridge, 'gauss', lasso, elasticnet, rvr_poly, xgb]
    model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly', 'xgb']

    model_para_list = [{'variancethreshold__threshold': var_threshold, 'elasticnet__random_state': rand_seed,
                        'elasticnet__n_jobs': n_jobs},

                       {'variancethreshold__threshold': var_threshold, 'rf__n_estimators': 500, 'rf__criterion': 'mse',
                        'rf__max_features': 0.33, 'rf__min_samples_leaf': 5, 'rf__n_jobs':n_jobs,
                        'rf__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'rvr__kernel': 'linear',
                        'rvr__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold,
                        'kernelridge__alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
                        'kernelridge__kernel': 'polynomial', 'kernelridge__degree': [1, 2], 'cv': 5,
                        'search_params': {'n_jobs': n_jobs}},

                       {'variancethreshold__threshold': var_threshold,
                        'gauss__kernel': gp.kernels.RBF(10.0, (1e-7, 10e7)), 'gauss__n_restarts_optimizer': 100,
                        'gauss__normalize_y': True, 'gauss__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'elasticnet__random_state': rand_seed,
                        'elasticnet__n_jobs': n_jobs},

                       {'variancethreshold__threshold': var_threshold, 'elasticnet__random_state': rand_seed,
                        'elasticnet__n_jobs': n_jobs},

                       {'variancethreshold__threshold': var_threshold, 'rvr__kernel': 'poly', 'rvr__degree': 1,
                        'rvr__random_state': rand_seed},

                        {'variancethreshold__threshold': var_threshold, 'xgboostadapted__n_jobs': 1,
                         'xgboostadapted__max_depth': [1, 2, 3, 6, 8], 'xgboostadapted__n_estimators': 100,
                         'xgboostadapted__reg_alpha': [0.0001, 0.01, 0.1, 1, 10],
                         'xgboostadapted__reg_lambda': [0.0001, 0.01, 0.1, 1, 10, 20],
                         'xgboostadapted__random_seed': rand_seed, 'search_params': {'n_jobs': n_jobs}}]

    for ind in range(0, len(model_required)):  # run only for required models and not all
        print('model required index and name:', ind, model_required[ind])

        i = model_names.index(model_required[ind])  # find index of required model in model_names list and use this index i to access model params
        assert model_required[ind] == model_names[i] # sanity check
        print('model picked from the list', model_names[i], model_list[i], '\n')

        if confounds is None:
            if pca_status:
                preprocess_X = ['variancethreshold', 'zscore', pca]
            else:
                preprocess_X = ['variancethreshold', 'zscore']
        else:
            if pca_status:
                preprocess_X = ['variancethreshold', 'zscore', 'remove_confound', pca]
            else:
                preprocess_X = ['variancethreshold', 'zscore', 'remove_confound']

        print('Preprocessing includes:', preprocess_X)

        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed).split(data_df,
                                                                                                           data_df.bins)

        scores, model = run_cross_validation(X=X, y=y, data=data_df, preprocess_X=preprocess_X, confounds=confounds,
                                             problem_type='regression', model=model_list[i], cv=cv,
                                     return_estimator='all', model_params=model_para_list[i], seed=rand_seed,
                                             scoring=
                                     ['neg_mean_absolute_error', 'neg_mean_squared_error','r2'], n_jobs=n_jobs) # adapted run_cross_validation to give n_jobs

        scores_cv[model_names[i]] = scores

        if model_names[i] == 'kernel_ridge' or model_names[i] == 'xgb':
            models[model_names[i]] = model.best_estimator_
            print('best model', model.best_estimator_)
            print('best para', model.best_params_)
        else:
            models[model_names[i]] = model
            print('best model', model)

        # # get predictions for test data
        # y_true = test_df[y_test]
        # y_pred = model.predict(test_df[X_test]).ravel()
        # y_delta = y_true - y_pred
        # print(y_true.shape, y_pred.shape)
        # mae = round(mean_absolute_error(y_true, y_pred), 3)
        # mse = round(mean_squared_error(y_true, y_pred), 3)
        # corr = round(np.corrcoef(y_pred, y_true)[1, 0], 3)
        # print('----------', mae, mse, corr)
        # output_df[model_names[i]] = y_pred
        #
        # results[model_names[i]] = {'predictions': y_pred, 'true': y_true,
        #                                        'delta': y_delta, 'mae': mae, 'mse': mse, 'corr': corr}
        # print(results)

    print('ALL DONE')

    # pickle.dump(results, open(output_dir / f'{output_filenm[1]}_{model_names[i]}.results', "wb"))
    # pickle.dump(scores_cv, open(output_dir / f'{output_filenm[1]}_{model_names[i]}.scores', "wb"))
    # pickle.dump(models, open(output_dir / f'{output_filenm[1]}_{model_names[i]}.models', "wb"))
    # output_df.to_csv(output_dir / f'{output_filenm[1]}_{model_names[i]}_prediction.csv', index=False)

    # print('Output file name')
    # print(output_dir / f'{output_filenm[1]}_{model_names[i]}.results')


    # pickle.dump(results, open(f'{output_path}.{model_names[i]}.results', "wb"))
    pickle.dump(scores_cv, open(f'{output_path}.{model_names[i]}.scores', "wb"))
    pickle.dump(models, open(f'{output_path}.{model_names[i]}.models', "wb"))
    # output_df.to_csv(f'{output_path}.{model_names[i]}_prediction.csv', index=False)

    # print('Output file name')
    # print(f'{output_path}.{model_names[i]}.results')

    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print("--- %s hours ---" % ((time.time() - start_time)/3600))










