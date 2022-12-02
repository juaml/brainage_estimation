import os
import argparse
import pickle
from pathlib import Path
from brainage import calculate_voxelwise_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, help="path to features dir")  # eg '../data/ADNI'
    # parser.add_argument("--output_path", type=str, help="path to output_dir")  # eg'../results/ADNI'
    parser.add_argument("--subject_filepaths", type=str, help="path to csv or txt file with subject filepaths") # eg: '../data/ADNI/ADNI_paths_cat12.8.csv'
    parser.add_argument("--output_prefix", type=str, help="prefix added to features filename ans results (predictions) file name") # eg: 'ADNI'
    parser.add_argument("--mask_file", type=str, help="path to GM mask nii file",
                        default='../masks/brainmask_12.8.nii')
    parser.add_argument("--smooth_fwhm", type=int, help="smoothing FWHM", default=4)
    parser.add_argument("--resample_size", type=int, help="resampling kernel size", default=4)

    # python3 calculate_features_voxelwise.py --features_path ../data/ixi/ --subject_filepaths ../data/ixi/ixi_paths_cat12.8.csv --output_prefix ixi --mask_file ../masks/brainmask_12.8.nii --smooth_fwhm 4 --resample_size 8
    
    # example inputs
    # features_path = Path('../data/ixi/')
    # subject_filepaths = '../data/ixi_paths_cat12.8.csv'
    # output_prefix = 'ixi'
    # mask_file = '../masks/brainmask_12.8.nii'
    # smooth_fwhm = 4
    # resample_size = 8

    args = parser.parse_args()
    features_path = Path(args.features_path)
    subject_filepaths = args.subject_filepaths
    output_prefix = args.output_prefix
    mask_file = args.mask_file
    smooth_fwhm = args.smooth_fwhm
    resample_size = args.resample_size

    print('Subjects filepaths: ', subject_filepaths)
    print('Directory to features path: ',  features_path)
    print('Results filename prefix: ', output_prefix)
    print('GM mask used: ', mask_file)
    print('smooth_fwhm:', smooth_fwhm)
    print('resample_size:', resample_size, '/n')

    data_resampled = calculate_voxelwise_features(subject_filepaths, mask_file, smooth_fwhm=smooth_fwhm, resample_size=resample_size)

    features_path.mkdir(exist_ok=True, parents=True)

    full_filename = str(output_prefix) + '.S' + str(smooth_fwhm) + '_R' + str(resample_size)
    filename = os.path.join(features_path, full_filename)
    print('filename for features created: ', filename)
    pickle.dump(data_resampled, open(filename, "wb"), protocol=4)
    data_resampled.to_csv(filename + '.csv', index=False)
