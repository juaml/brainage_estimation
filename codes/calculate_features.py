#!/home/smore/.venvs/py3smore/bin/python3
import os.path
from pathlib import Path
import argparse
from brainage import read_sub_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="Output file path")
    parser.add_argument("output_filenm", type=str, help="Output file name")
    parser.add_argument("smooth_fwhm", type=int, help="smooth_fwhm")
    parser.add_argument("resample_flag", type=int, help="resampling flag 0 or 1")
    parser.add_argument("resample_size", type=int, help="resample_size")
    parser.add_argument("phenotype_file", type=str, help="Phenotype file name")

    args = parser.parse_args()
    output_path = Path(args.output_path)
    output_filenm = args.output_filenm
    smooth_fwhm = args.smooth_fwhm
    resample_flag = bool(args.resample_flag)
    resample_size = args.resample_size
    phenotype_file = args.phenotype_file
    mask_dir = '../masks/brainmask_12.8.nii'

    # example inputs
    # output_path = Path('../data/ixi/')
    # output_filenm = 'ixi'
    # smooth_fwhm = 0
    # resample_flag = bool(1)
    # resample_size = 8
    # phenotype_file = '/data/project/brainage/brainage_julearn_final/data_new/ixi/ixi_paths_cat12.8.csv'
    # mask_dir = '/data/project/brainage/brainage_julearn_final/masks/brainmask_12.8.nii'
    
    print('output_path:', output_path)
    print('output_filenm:', output_filenm)
    print('smooth_fwhm:', smooth_fwhm)
    print('resample_flag:', resample_flag)
    print('resample_size:', resample_size)
    print('phenotype_file:', phenotype_file)
    print('mask_dir:', mask_dir, '/n')

    data_resampled = read_sub_data(phenotype_file, mask_dir, smooth_fwhm=smooth_fwhm,
                                         resample_flag=resample_flag, resample_size=resample_size)

    output_path.mkdir(exist_ok=True, parents=True)

    full_filename = str(output_filenm) + '_S' + str(smooth_fwhm) + '_R' + str(resample_size)
    filename = os.path.join(output_path, full_filename)
    print('filename for features created: ', filename)
    pickle.dump(data_resampled, open(filename, "wb"), protocol=4)
    data_resampled.to_csv(filename + '.csv', index=False)
