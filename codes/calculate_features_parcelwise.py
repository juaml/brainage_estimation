import os
import pickle
import argparse
from pathlib import Path
from brainage import calculate_parcelwise_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, help="path to features dir")  # eg '../data/ADNI'
    # parser.add_argument("--output_path", type=str, help="path to output_dir")  # eg'../results/ADNI'
    parser.add_argument("--subject_filepaths", type=str, help="path to csv or txt file with subject filepaths") # eg: '../data/ADNI/ADNI_paths_cat12.8.csv'
    parser.add_argument("--output_prefix", type=str, help="prefix added to features filename ans results (predictions) file name") # eg: 'ADNI'
    parser.add_argument("--mask_file", type=str, help="path to mask nii file")
    parser.add_argument("--num_parcels", type=str, help="Number of parcels")

    # python3 calculate_features_parcelwise.py --features_path ../data/ixi/ --subject_filepaths ../data/ixi/ixi_paths_cat12.8.csv --output_prefix ixi --mask_file ../masks/BSF_173.nii --num_parcels 173
   
    # example inputs
    # features_path = Path('../data/ixi/')
    # subject_filepaths = '../data/ixi_paths_cat12.8.csv'
    # output_prefix = 'ixi'
    # mask_file = '../masks/BSF_173.nii'
    # num_parcels = 173

    args = parser.parse_args()
    features_path = Path(args.features_path)
    subject_filepaths = args.subject_filepaths
    output_prefix = args.output_prefix
    mask_file = args.mask_file
    num_parcels = args.num_parcels

    print('Subjects filepaths: ', subject_filepaths)
    print('Directory to features path: ',  features_path)
    print('Results filename prefix: ', output_prefix)
    print('GM mask used: ', mask_file)
    print('Number of parcels:', num_parcels, '/n')
    
    data_parcels = calculate_parcelwise_features(subject_filepaths, mask_file, num_parcels)

    features_path.mkdir(exist_ok=True, parents=True)
    
    full_filename = str(output_prefix) + '.' + str(num_parcels)
    filename = os.path.join(features_path, full_filename)
    print('filename for features created: ', filename)
    pickle.dump(data_parcels, open(filename, "wb"), protocol=4)
    data_parcels.to_csv(filename + '.csv', index=False)