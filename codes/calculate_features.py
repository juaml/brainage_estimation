#!/home/smore/.venvs/py3smore/bin/python3
import os.path
from pathlib import Path
import argparse
import pickle
from brainage import read_sub_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, help="Output file path")
    parser.add_argument("--output_filenm", type=str, help="Output file name")
    parser.add_argument("--smooth_fwhm", type=int, help="smooth_fwhm")
    parser.add_argument("--resample_size", type=int, help="resample_size")
    parser.add_argument("--phenotype_file", type=str, help="Phenotype file name")
    parser.add_argument("--mask_dir", type=str, help="Phenotype file name", default= '../masks/brainmask_12.8.nii')


    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_filenm = args.output_filenm
    smooth_fwhm = args.smooth_fwhm
    resample_size = args.resample_size
    phenotype_file = args.phenotype_file
    mask_dir = '../masks/brainmask_12.8.nii'

    # example inputs
    # output_path = Path('../data/ADNI/')
    # output_filenm = 'ADNI'
    # smooth_fwhm = 4
    # resample_size = 8
    # phenotype_file = '../data/ADNI_paths_cat12.8.csv'
    # mask_dir = '../masks/brainmask_12.8.nii'
    
    print('output_path:', output_folder)
    print('output_filenm:', output_filenm)
    print('smooth_fwhm:', smooth_fwhm)
    print('resample_size:', resample_size)
    print('phenotype_file:', phenotype_file)
    print('mask_dir:', mask_dir, '/n')

    data_resampled = read_sub_data(phenotype_file, mask_dir, smooth_fwhm=smooth_fwhm, resample_size=resample_size)

    output_folder.mkdir(exist_ok=True, parents=True)

    full_filename = str(output_filenm) + '_S' + str(smooth_fwhm) + '_R' + str(resample_size)
    filename = os.path.join(output_folder, full_filename)
    print('filename for features created: ', filename)
    pickle.dump(data_resampled, open(filename, "wb"), protocol=4)
    data_resampled.to_csv(filename + '.csv', index=False)
