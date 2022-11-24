#from read_data_mask_resampled import *
from brainage import read_sub_data
from julearn.transformers import register_transformer
from pathlib import Path
import pandas as pd
import argparse
import pickle
import os
import re


def model_pred(test_df, model_file, feature_space_str):
    """This functions predicts age
    Args:
        test_df (dataframe): test data
        model_file (pickle file): trained model file
        feature_space_str (string): feature space name

    Returns:
        dataframe: predictions from the model
    """    

    model = pickle.load(open(model_file, 'rb')) # load model
    pred = pd.DataFrame()
    for key, model_value in model.items():
        X = data_df.columns.tolist()
        pre_X, pre_X2 = model_value.preprocess(test_df[X], test_df[X])  # preprocessed data
        y_pred = model_value.predict(test_df).ravel()
        print(y_pred.shape)
        pred[feature_space_str + '+' + key] = y_pred
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_path", type=str, help="path to features dir")  # eg '../data/ADNI'
    parser.add_argument("--subject_filepaths", type=str, help="path to csv or txt file with subject filepaths") # eg: '../data/ADNI/ADNI_paths_cat12.8.csv'
    parser.add_argument("--output_path", type=str, help="path to output_dir")  # eg'../results/ADNI'
    parser.add_argument("--output_prefix", type=str, help="results file name prefix") # eg: 'ADNI'
    parser.add_argument("--mask_file", type=str, help="path to GM mask nii file",
                        default='../masks/brainmask_12.8.nii')
    parser.add_argument("--smooth_fwhm", type=int, help="smoothing FWHM", default=4)
    parser.add_argument("--resample_size", type=int, help="resampling kernel size", default=4)
    parser.add_argument("--model_file", type=str, help="Trained model to be used to predict",
                        default='../trained_models/4sites.S4_R4_pca.gauss.models')
    # For testing
    # python3 predict_age.py --features_path ../data/ADNI --subject_filepaths ../data/ADNI/ADNI_paths_cat12.8.csv --output_path ../results/ADNI --output_prefix ADNI --mask_file ../masks/brainmask_12.8.nii  --smooth_fwhm 4 --resample_size 4 --model_file ../trained_models/4sites.S4_R4_pca.gauss.models

    args = parser.parse_args()
    features_path = Path(args.features_path)
    subject_filepaths = args.subject_filepaths
    output_path = Path(args.output_path)
    output_prefix = args.output_prefix
    smooth_fwhm = args.smooth_fwhm
    resample_size = args.resample_size
    mask_file = args.mask_file
    model_file = args.model_file

    print('\nBrain-age trained model used: ', model_file)
    print('Subjects filepaths (test data): ', subject_filepaths)
    print('saved features path: ',  features_path)
    print('Results directory: ', output_path)
    print('Results filename: ', output_prefix)
    print('GM mask used: ', mask_file)

    # get feature space name from the model file entered and
    # create feature space name using the input values (smoothing, resampling)
    # match them: they should be same

    # get feature space name from the model file
    pipeline_name1 = model_file.split('/')[-1]
    feature_space = pipeline_name1.split('.')[1]
    model_name = pipeline_name1.split('.')[2]
    pipeline_name = feature_space + '.' + model_name
    
    # create feature space name using the input values (smoothing, resampling)
    pca_string = re.findall(r"pca", feature_space)
    if len(pca_string) == 1:
        feature_space_str = 'S' + str(smooth_fwhm) + '_R' + str(resample_size) + '_pca'
    else:
        feature_space_str = 'S' + str(smooth_fwhm) + '_R' + str(resample_size)

    # match them: they should be same
    assert(feature_space_str == feature_space), f"Mismatch in feature parameters entered ({feature_space_str}) & features used for model training ({feature_space})"

    print('Feature space: ', feature_space)
    print('Model name: ', model_name)

    # Create directories, create features if they don't exists
    output_path.mkdir(exist_ok=True, parents=True)
    features_path.mkdir(exist_ok=True, parents=True)
    features_filename = str(output_prefix) + '_S' + str(smooth_fwhm) + '_R' + str(resample_size)
    features_fullfile = os.path.join(features_path, features_filename)
    print('\nfilename for features created: ', features_fullfile)

    if os.path.isfile(features_fullfile): # check if features file exists
        print('\n----File exists')
        data_df = pickle.load(open(features_fullfile, 'rb'))
        print('Features loaded')
    else:
        print('\n-----Extracting features')
        # create features
        data_df = read_sub_data(subject_filepaths, mask_file, smooth_fwhm=smooth_fwhm, resample_size=resample_size)
        # save features
        pickle.dump(data_df, open(features_fullfile, "wb"), protocol=4)
        data_df.to_csv(features_fullfile + '.csv', index=False)
        print('Feature extraction done and saved')

    # get predictions and save
    try:
        predictions_df = model_pred(data_df, model_file, feature_space_str)
        # save predictions
        predictions_filename = str(output_prefix) + '_' + pipeline_name + '_prediction.csv'
        predictions_fullfile = os.path.join(output_path, predictions_filename)
        print('\nfilename for predictions created: ', predictions_fullfile)
        predictions_df.to_csv(predictions_fullfile, index=False)

        print(predictions_df)

    except FileNotFoundError:
        print(f'{model_file} is not present')




