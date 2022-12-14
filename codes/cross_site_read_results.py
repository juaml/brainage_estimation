import pickle
import os.path
import argparse
import pandas as pd

# all possible inputs
## cross site (3 sites)
# data_nm = '..results/camcan_enki_1000brains/camcan_enki_1000brains_'
# data_nm = '..results/ixi_enki_1000brains/ixi_enki_1000brains_'
# data_nm = '..results/ixi_camcan_enki/ixi_camcan_enki_'
# data_nm = '..results/ixi_camcan_1000brains/ixi_camcan_1000brains_'
## cross-site (4 sites)
# data_nm = '..results/ixi_camcan_enki_1000brains/4sites_'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_nm", type=str, help="Output path for one dataset")

    args = parser.parse_args()
    data_nm = args.data_nm

    # Filename to save results
    cv_file_ext = 'cv_scores.csv'
    cv_file_ext_selected = 'cv_scores_selected.csv'

    # Complete results filepaths
    cv_filename = data_nm + cv_file_ext
    cv_filename_selected = data_nm + cv_file_ext_selected


    # all model names
    model_names = ['lin_reg', 'ridge', 'rf', 'rvr_lin', 'kernel_ridge', 'gauss', 'lasso', 'elasticnet', 'rvr_poly'] #'xgb'
    model_names_new = ['LiR', 'RR', 'RFR', 'RVRlin', 'KRR', 'GPR', 'LR', 'ENR', 'RVRpoly'] # 'XGB'

    # all feature spaces names
    data_list = ['173', '473', '873','1273', 'S0_R4', 'S0_R4_pca', 'S4_R4', 'S4_R4_pca', 'S8_R4', 'S8_R4_pca',
                       'S0_R8', 'S0_R8_pca', 'S4_R8', 'S4_R8_pca', 'S8_R8', 'S8_R8_pca']
    data_list_new = ['173', '473', '873','1273', 'S0_R4', 'S0_R4 + PCA', 'S4_R4', 'S4_R4 + PCA', 'S8_R4', 'S8_R4 + PCA',
                       'S0_R8', 'S0_R8 + PCA', 'S4_R8', 'S4_R8 + PCA', 'S8_R8', 'S8_R8 + PCA']

    # check which scores file is missing
    missing_outs = []
    for data_item in data_list:
        for model_item in model_names:
            scores_item = data_nm + data_item + '.' + model_item + '.scores' # create the complete path to scores file
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
            scores_item = data_nm + data_item + '.' + model_item + '.scores'  # create the complete path to scores file
            if os.path.isfile(scores_item):
                res = pickle.load(open(scores_item, 'rb'))
                df = pd.DataFrame()
                mae_all, mse_all, corr_all, corr_delta_all, key_all = list(), list(), list(), list(), list()
                for key, value in res.items():
                    mae = round(value['test_neg_mean_absolute_error'].mean() * -1, 3)
                    mse = round(value['test_neg_mean_squared_error'].mean() * -1, 3)
                    corr = round(value['test_r2'].mean(), 3)
                    mae_all.append(mae)
                    mse_all.append(mse)
                    corr_all.append(corr)
                    key_all.append(key)

                df['model'] = key_all
                df['data'] = len(mae_all) * [data_item]
                df['cv_mae'] = mae_all
                df['cv_mse'] = mse_all
                df['cv_corr'] = corr_all
                # print(df)
                df_cv = pd.concat([df_cv, df], axis=0)

    df_cv = df_cv.reset_index(drop=True)
    df_cv['workflow_name'] = df_cv['data'] + ' + ' + df_cv['model']
    df_cv['data'] = df_cv['data'].replace(data_list, data_list_new)
    df_cv['model'] = df_cv['model'].replace(model_names, model_names_new)
    df_cv['workflow_name_updated'] = df_cv['data'] + ' + ' + df_cv['model']
    df_cv.reset_index(drop=True, inplace=True)

    # selected 32 workflows (since we have more then 32 selected workflows)
    selected_workflows_df = pd.DataFrame([
        '173 + GPR',
        '473 + LR',
        '473 + RVRpoly',
        '1273 + GPR',
        'S4_R4 + RR',
        'S4_R4 + GPR',
        'S4_R4 + PCA + RFR',
        'S4_R4 + PCA + RVRlin',
        'S8_R4 + PCA + RVRlin',
        'S8_R4 + PCA + GPR',
        'S0_R8 + PCA + ENR',
        'S0_R8 + PCA + RVRpoly',
        'S4_R8 + RR',
        'S8_R8 + RR',
        'S8_R8 + KRR',
        'S8_R8 + PCA + ENR',
        'S4_R4 + PCA + GPR',
        'S4_R4 + RVRlin',
        'S4_R4 + PCA + RR',
        'S4_R8 + RVRlin',
        'S8_R4 + KRR',
        'S0_R4 + LR',
        'S8_R4 + PCA + RVRpoly',
        'S0_R8 + RVRpoly',
        'S4_R8 + LR',
        '873 + GPR',
        'S8_R4 + PCA + LR',
        '1273 + RVRpoly',
        '873 + ENR',
        '173 + LR',
        'S0_R8 + PCA + LR',
        '173 + RFR'], columns=['workflow_name_updated'])

    df_final = df_cv.merge(selected_workflows_df, how='inner', on=['workflow_name_updated'])

    # save the csv files
    print('\n cv results file:', cv_filename)
    print(df_cv)
    print('\n selected results file:', cv_filename_selected)
    print(df_final)
    df_cv.to_csv(cv_filename, index=False)
    df_final.to_csv(cv_filename_selected, index=False)


    # # check model parameters
    print('\n Model Parameters')
    error_models = list()
    for data_item in data_list:
        for model_item in model_names:
            model_item = data_nm + data_item + '.' + model_item + '.models'  # get models
            # print('\n','model filename', model_item)
            if os.path.isfile(model_item):
                print('\n', 'model filename', model_item)
                res = pickle.load(open(model_item, 'rb'))
                # print(res)
                for key, value in res.items():
                    print(key)

                    if key == 'gauss':
                        model = res['gauss']['gauss']
                        # print(model.get_params())
                        print(model.kernel_.get_params())

                    elif key == 'kernel_ridge':
                        model = res['kernel_ridge']['kernelridge']
                        print(model)
                        # print(model.get_params())

                    elif key == 'rvr_lin':
                        model = res['rvr_lin']['rvr']
                        print(model)
                        # print(model.get_params())

                    elif key == 'rvr_poly':
                        model = res['rvr_poly']['rvr']
                        print(model)
                        # print(model.get_params())

                    elif key == 'rf':
                        model = res['rf']['rf']
                        print(model)
                        # print(model.get_params())

                    else:
                        model = res[key]['elasticnet']
                        # print(model.get_params())
                        print(model.lambda_best_)

            else:
                error_models.append(model_item)
