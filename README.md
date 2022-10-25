1.  **Create virtual enviornment**

    `cd brainage_estimation`  
    `python3 -m venv test_package_env`  
    `source test_package_env/bin/activate`   
    `pip install -r requirements.txt`  
    `pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip`   
    `brew install gcc` (For mac users)  
    `pip install glmnet`  


2. To run codes: `cd codes`

    1. **To get predictions**  
    
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
    `python3 calculate_features.py ../data/ADNI/ ADNI 4 1 8 ../data/ADNI_paths_cat12.8.csv ../masks/brainmask_12.8.nii`
    
    
    3. **Within-site: Train models**
    `python3 within_site_train.py --demo_path ../data/ixi/ixi_subject_list_cat12.8.csv --data_path ../data/ixi/ixi_173 --output_filenm ixi/ixi_173 --models ridge --pca_status 0`
    or 
    `condor_submit within_site_ixi.submit`


    4. **Within-site: Read results from saved models**
    `python3 within_site_read_results.py --data_nm /ixi/ixi_`


    5. **Within-site: Get predictions from 128 workflows**
    `python3 within_site_combine_predictions.py --data_nm /ixi/ixi_`


    6. **Within site: Bias correction**
    `python3 within_site_bias_correction.py --dataset_flag ixi`


    7. **Cross-site: Train models**
      `python3 cross_site_train.py --data_path ../data/ixi_camcan_enki/ixi_camcan_enki_173 --output_path ../results/ixi_camcan_enki/ixi_camcan_enki_173 --   models rvr_lin --confounds None --pca_status 0 --n_jobs 5`

      `python3 cross_site_train.py --data_path ../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_173 --output_path ../results/ixi_camcan_enki_1000brains/4sites_173 --models rvr_lin --confounds None --pca_status 0 --n_jobs 5`


    8. **Cross-site: Read results from saved models**
      `python3 cross_site_read_results.py --data_nm /ixi_camcan_enki/ixi_camcan_enki_`
      `python3 cross_site_read_results.py --data_nm /ixi_camcan_enki_1000brains/4sites_ `


    9. **Cross-site: Get predictions from workflows**
      `python3 cross_site_combine_predictions.py --model_folder /ixi_camcan_enki/ixi_camcan_enki_ --test_data_name /1000brains/1000brains_ --save_file_ext pred_1000brains_all`
      `python3 cross_site_combine_predictions.py --model_folder /ixi_camcan_enki_1000brains/4sites_ --test_data_name /ADNI/ADNI_ --save_file_ext pred_adni_all`
   
   
   10. **Cross site: Bias correction**
    `python3 cross_site_bias_correction.py --data_path" ../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_S4_R4' --output_filenm ixi_camcan_enki_1000brains/4sites_S4_R4_pca_cv.gauss --mod_nm gauss --confounds None`
