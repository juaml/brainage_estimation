1. **Folder Structure**
There are following folders and files:
   
   1. `trained_models`: contains 10 trained models. Models are trained using voxel-wise GM images (from CAT 12.8) with additional smoothing (S0, S4, S8) and resampling (R4, R8) as features with various ML algorithms
   
   2. `data`:
      1. `ADNI_mri.tar.gz`: contains preprocessed .nii files for all ADNI subjects (go to `data` folder and run `tar -zxvf ADNI_mri.tar.gz` to get the .nii files)
      2. `ADNI_paths_cat12.8.csv`: csv with path to .nii files of ADNI subjects
      3. `ADNI_demographics.csv`: csv with demographic information for ADNI subjects as dowloaded from website
   
   3. `codes`: contains two python script, details below.
   
   4. `requirements.txt`: contains list of python packages to be installed
   
   
2. **codes**
   
1. `predict_age.py`: This is main file for predicting age. It takes in 8 arguments: `features_path`: path to directory to save or load features(default=../data), `output_path`: path to results directory (default=../results ), `output_filenm`: results filename extension (default='ADNI'), `smooth_fwhm`: smoothing FWHM (default=4), `resample_size`: Resampling kernel size (default=4), `subject_filepaths`: Path to .csv or .txt file with test subject filepaths (default=../data/ADNI_paths_cat12.8.csv), `mask_dir`: Path to .nii file for the GM mask (default=../masks/brainmask_12.8.nii), `model_file`: Path to trained model (default=../trained_models/4sites_S4_R4_pca.gauss.models)
    
2. `train_within_site.py`: Train within-site models
    
3.  `train_cross_site.py`: Train cross-site models

4. `ixi.submit`

5. `ixi_camcan_enki.submit`
    
6. `4sites_bestmodel.submit`
   

3. **brainage** 

1. `read_data_mask_resampled.py`: called in predict_age.py to calculate features

2.  `create_splits.py`


3.  **Create virtual enviornment**
`cd brainage_estimation`
`python3 -m venv test_env`
`source test_env/bin/activate` 
`pip install -r requirements.txt`
`pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip` 
`brew install gcc` (For mac users)
`pip install glmnet`


1. **To get predictions**
`cd codes`

Example-1:
`python3 predict_age.py --features_path ../data --output_path ../results --output_filenm 'ADNI' --smooth_fwhm 4 --resample_size 8 --subject_filepaths ../data/ADNI_paths_cat12.8.csv --mask_dir ../masks/brainmask_12.8.nii --model_file ../trained_models/4sites_S4_R8.ridge.models`
It calculates features with 4mm smoothing and 8mm resampling (`S4_R8`) for subjects in list (`../data/ADNI_paths_cat12.8.csv`) and calculates predictions using S4_R8+ridge workflow (`../trained_models/4sites_S4_R8.ridge.models`)

Example-2:
`python3 predict_age.py --smooth_fwhm 4 --resample_size 4 --model_file ../trained_models/4sites_S4_R4_pca.rvr_lin.models`
Calculates predictions using `S4_R4_pca+rvr_lin` workflow

Example-3:
`python3 predict_age.py`
Calculates predictions using `S4_R4_pca+gauss` workflow

2. **calculate features**
3. **Train within-site models**
4. **Train cross-site models**
5. 


