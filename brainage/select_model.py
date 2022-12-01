import xgboost as xgb
from skrvm import RVR
from glmnet import ElasticNet
import sklearn.gaussian_process as gp
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from brainage import XGBoostAdapted
    
def define_models():
    # Define all models and model parameters
    rvr_linear = RVR()
    rvr_poly = RVR()
    kernel_ridge = KernelRidge()
    lasso = ElasticNet(alpha=1, standardize=False)
    elasticnet = ElasticNet(alpha=0.5, standardize=False)
    ridge = ElasticNet(alpha=0, standardize=False)
    xgb = XGBoostAdapted(early_stopping_rounds=10, eval_metric='mae', eval_set_percent=0.2)
    pca = PCA(n_components=None)  # max as many components as sample size


    model_list = [ridge, 'rf', rvr_linear, kernel_ridge, 'gauss', lasso, elasticnet, rvr_poly, xgb]
    model_para_list = [
                        {'variancethreshold__threshold': var_threshold, 'elasticnet__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'rf__n_estimators': 500, 'rf__criterion': 'mse',
                        'rf__max_features': 0.33, 'rf__min_samples_leaf': 5,
                        'rf__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'rvr__kernel': 'linear',
                        'rvr__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold,
                        'kernelridge__alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
                        'kernelridge__kernel': 'polynomial', 'kernelridge__degree': [1, 2], 'cv': 5},

                       {'variancethreshold__threshold': var_threshold,
                        'gauss__kernel': gp.kernels.RBF(10.0, (1e-7, 10e7)), 'gauss__n_restarts_optimizer': 100,
                        'gauss__normalize_y': True, 'gauss__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'elasticnet__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'elasticnet__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'rvr__kernel': 'poly', 'rvr__degree': 1,
                        'rvr__random_state': rand_seed},

                       {'variancethreshold__threshold': var_threshold, 'xgboostadapted__n_jobs': 1,
                        'xgboostadapted__max_depth': [1, 2, 3, 6, 8, 10, 12], 'xgboostadapted__n_estimators': 100,
                        'xgboostadapted__reg_alpha': [0.001, 0.01, 0.05, 0.1, 0.2],
                        'xgboostadapted__random_seed': rand_seed, 'cv': 5}]  # 'search_params':{'n_jobs': 5}
                        ]
                        
    return model_list, model_para_list
