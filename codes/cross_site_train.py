import time
import pickle
import argparse
import pandas as pd
from pathlib import Path

from brainage import read_data, XGBoostAdapted

import xgboost as xgb
from skrvm import RVR
from glmnet import ElasticNet
import sklearn.gaussian_process as gp
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RepeatedStratifiedKFold

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.transformers import register_transformer

start_time = time.time()

def none_or_str(value):
    if value == 'None':
        return None
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demographics_file", type=str, help="Demographics file path")
    parser.add_argument("--features_file", type=str, help="Features file path")
    parser.add_argument("--output_path", type=str, help="Path to output directory")
    parser.add_argument("--output_prefix", type=str, help="Output prefix (used {dataname}.{featurename}")
    parser.add_argument("--models", type=str, nargs='?', const=1, default="ridge",
                       help="models to use (comma seperated no space): ridge,rf,rvr_linear")
    parser.add_argument("--pca_status", type=int, default=0,
                       help="0: no pca, 1: yes pca")
    parser.add_argument("--confounds", type=none_or_str, help="confounds", default=None)
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run")

    configure_logging(level='INFO')

    # Parse the arguments
    args = parser.parse_args()
    demographics_file = args.demographics_file
    features_file = args.features_file
    output_path = Path(args.output_path)
    output_prefix = args.output_prefix
    model_required = [x.strip() for x in args.models.split(',')]  # converts string into list
    confounds = args.confounds
    pca_status = bool(args.pca_status)
    n_jobs = args.n_jobs
    output_path.mkdir(exist_ok=True, parents=True) # check and create output directory

    # initialize random seed and create test indices
    rand_seed = 200
    n_repeats = 5 # for inner CV
    n_splits = 5  # how many train and test splits (both for other and inner)

    print('\nDemographics file: ', demographics_file)
    print('Features file: ', features_file)
    print('Ouput path : ', output_path)
    print('Ouput prefix: ', output_prefix)
    print('Model:', model_required, type(model_required))
    print('PCA status : ', pca_status)
    print('Random seed : ', rand_seed)
    print('Num of splits for kfolds : ', n_splits, '\n')
    print('confounds:', confounds, type(confounds))
    print('Num of parallel jobs initiated: ', n_jobs, '\n')

    # read the features, demographics and define X and y
    data_df, X, y = read_data(features_file=features_file, demographics_file=demographics_file)

    # register VarianceThreshold as a transformer
    register_transformer('variancethreshold', VarianceThreshold, returned_features='unknown', apply_to='all_features')
    var_threshold = 1e-5

    # Initialize variables, set random seed, create classes for age
    scores_cv, models, results = {}, {}, {}
    qc = pd.cut(data_df['age'].tolist(), bins=5, precision=1)  # create bins for train data only
    print('age_bins', qc.categories, 'age_codes', qc.codes)
    data_df['bins'] = qc.codes # add bin/classes as a column in train df

    # Define all models and model parameters
    rvr_linear = RVR()
    rvr_poly = RVR()
    kernel_ridge = KernelRidge()  # kernelridge
    lasso = ElasticNet(alpha=1, standardize=False)
    elasticnet = ElasticNet(alpha=0.5, standardize=False)
    ridge = ElasticNet(alpha=0, standardize=False)
    xgb = XGBoostAdapted(early_stopping_rounds=10, eval_metric='mae', eval_set_percent=0.2)
    pca = PCA(n_components=None)  # max as many components as sample size

    model_names = ['ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly', 'xgb']
    model_list = [ridge, 'rf', rvr_linear, kernel_ridge, 'gauss', lasso, elasticnet, rvr_poly, xgb]

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

    # Define processing for X (features)
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
     
    # Get the model, its parameters, pca status and train
    for ind in range(0, len(model_required)):  # run only for required models and not all
        print('model required index and name:', ind, model_required[ind])
        i = model_names.index(model_required[ind])  # find index of required model in model_names list and use this index i to access model params
        assert model_required[ind] == model_names[i] # sanity check
        print('model picked from the list', model_names[i], model_list[i], '\n')
       
        # initialize dictionaries to save scores and models here to save every model separately
        scores_cv, models = {}, {}
        
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed).split(data_df, data_df.bins)

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

        print('Output file name')
        print(output_path / f'{output_prefix}.{model_names[i]}.models')
        pickle.dump(models, open(output_path / f'{output_prefix}.{model_names[i]}.models', "wb"))
        pickle.dump(scores_cv, open(output_path / f'{output_prefix}.{model_names[i]}.scores', "wb"))

    print('ALL DONE')
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print("--- %s hours ---" % ((time.time() - start_time)/3600))










