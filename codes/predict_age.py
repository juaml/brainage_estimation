from read_data_mask_resampled import *
from julearn.transformers import register_transformer
from pathlib import Path
import pandas as pd
import argparse
import pickle
import os.path
import os
import re


def model_pred(test_df, model_file, file_name):
    """Get predictions"""
    model = pickle.load(open(model_file, 'rb')) # load model
    pred = pd.DataFrame()
    for key, model_value in model.items():
        X = data_df.columns.tolist()
        pre_X, pre_X2 = model_value.preprocess(test_df[X], test_df[X]) # preprocessed data
        y_pred = model_value.predict(test_df).ravel()
        print(y_pred.shape)
        pred[file_name + '+' + key] = y_pred
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, help="Features file path", default='../data')  # features directory
    parser.add_argument("--output_path", type=str, help="Output file path", default='../results')  # results directory
    parser.add_argument("--output_filenm", type=str, help="Output file name", default='ADNI') # results filename extension
    parser.add_argument("--smooth_fwhm", type=int, help="smooth_fwhm", default=4) # smoothing FWHM
    parser.add_argument("--resample_size", type=int, help="resample_size", default=4) # Resampling kernel size
    parser.add_argument("--subject_filepaths", type=str, help="Subject filepath list", default='../data/ADNI_paths_cat12.8.csv')  # Path to .csv or .txt file with subject filepaths
    parser.add_argument("--mask_dir", type=str, help="GM mask", default='../masks/brainmask_12.8.nii') # Path .nii file for the GM mask
    parser.add_argument("--model_file", type=str, help="Final model to be used to predict",
                        default='../trained_models/4sites_S4_R4_pca.gauss.models')  # Trained model

    args = parser.parse_args()
    output_path = Path(args.output_path)
    features_path = Path(args.features_path)
    output_filenm = args.output_filenm
    smooth_fwhm = args.smooth_fwhm
    resample_size = args.resample_size
    subject_filepaths = args.subject_filepaths
    mask_dir = args.mask_dir
    model_file = args.model_file

    print('\nBrain-age trained model used: ', model_file)
    print('Subjects filepaths (test data): ', subject_filepaths)
    print('saved features path: ',  features_path)
    print('Results directory: ', output_path)
    print('Results filename: ', output_filenm)
    print('GM mask used: ', mask_dir)

    # get pipeline name (specifically feature name) and check feature parameters(smoothing & resampling, PCA or no PCA)
    # entered by user: they should match
    pipeline_name1 = model_file.split('/')[-1]
    pipeline_name = pipeline_name1[7:]
    feature_space = pipeline_name.split('.')[0]
    model_name = pipeline_name.split('.')[1]
    pca_string = re.findall(r"pca", pipeline_name)
    if len(pca_string) == 1:
        feature_space_str = 'S' + str(smooth_fwhm) + '_R' + str(resample_size) + '_pca'
    else:
        feature_space_str = 'S' + str(smooth_fwhm) + '_R' + str(resample_size)

    assert(feature_space_str == feature_space), "Mismatch in feature parameters entered and the features used for model training "

    print('Feature space: ', feature_space)
    print('Model name: ', model_name)

    # Create directories, create features if they don't exists
    output_path.mkdir(exist_ok=True, parents=True)
    features_path.mkdir(exist_ok=True, parents=True)
    features_filename = str(output_filenm) + '_S' + str(smooth_fwhm) + '_R' + str(resample_size)
    features_fullfile = os.path.join(features_path, features_filename)
    print('\nfilename for features created: ', features_fullfile)

    if os.path.isfile(features_fullfile): # check if features file exists
        print('\n----File exists')
        data_df = pickle.load(open(features_fullfile, 'rb'))
        print('Features loaded')
    else:
        print('\n-----Extracting features')
        # create features
        data_df = read_sub_data(subject_filepaths, mask_dir, smooth_fwhm=smooth_fwhm, resample_size=resample_size)
        # save features
        pickle.dump(data_df, open(features_fullfile, "wb"), protocol=4)
        data_df.to_csv(features_fullfile + '.csv', index=False)
        print('Feature extraction done and saved')

    # get predictions and save
    try:
        predictions_df = pd.DataFrame()
        y_pred1 = model_pred(data_df, model_file, feature_space_str)
        predictions_df = pd.concat([predictions_df, y_pred1], axis=1)

        # save predictions
        predictions_filename = str(output_filenm) + '_' + pipeline_name + '_prediction.csv'
        predictions_fullfile = os.path.join(output_path, predictions_filename)
        print('\nfilename for predictions created: ', predictions_fullfile)
        predictions_df.to_csv(predictions_fullfile, index=False)

    except FileNotFoundError:
        print(f'{model_file} is not present')




