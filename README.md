1.  **Set up**

    ```
    git clone https://github.com/juaml/brainage_estimation.git
    cd brainage_estimation
    python3 -m venv brainage_env
    source brainage_env/bin/activate
    pip install -r requirements.txt
    # install other packages
    pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
    #brew install gcc # for Mac users in case you don't have it
    pip install glmnet
    ```


2. **Run codes**

   `cd codes`

    1. **To get predictions**  
    
        Example-1: Predict age from CAT12.8 files.
        `python3 predict_age.py --features_path path_to_features_dir \                        
            --subject_filepaths path_to_txt_file \          
            --output_path path_to_output_dir \
            --output_filenm 'ADNI' \
            --mask_dir ../masks/brainmask_12.8.nii \
            --smooth_fwhm 4 --resample_size 4 \
            --model_file ../trained_models/4sites_S4_R4_pca.gauss.models`
            
        The arguments are:
        - `--features_path` should point to a directory where calculated features are stored as a `pickle` file.
        - `--subject_filepaths` should point to a text file containing path to the CAT12.8's `mwp1` file for each subject per line.
        - `--output_path` points to a directory where the predictions will be saved.
        - `--output_filenm` prefix for the output files.
        - `--mask_dir` should point to the GM mask used (defaults to `../masks/brainmask_12.8.nii`)
        - `--smooth_fwhm` smoothing kernel size to be used (defaults to 4)
        - `--resample_size` resampling of the voxels to isometric size (defaults to `4`)
        - `--model_file` should point to an already trained model (defaults to `4sites_S4_R4_pca.gauss.models`)
             
        This example will calculate features with 4mm smoothing and 4mm resampling (`S4_R4`) for all subjects in the provided list (via `--subject_filepaths`) and calculate predictions using the S4_R4_pca+gauss model. Note that if the features are available in the `--features_path` then they will not be recalculated.

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
      `python3 cross_site_train.py --data_path ../data/ixi_camcan_enki/ixi_camcan_enki_173 --output_path ../results/ixi_camcan_enki/ixi_camcan_enki_173 --models rvr_lin --confounds None --pca_status 0 --n_jobs 5`  

        `python3 cross_site_train.py --data_path ../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_173 --output_path ../results/ixi_camcan_enki_1000brains/4sites_173 --models rvr_lin --confounds None --pca_status 0 --n_jobs 5`


    8. **Cross-site: Read results from saved models**  
      `python3 cross_site_read_results.py --data_nm /ixi_camcan_enki/ixi_camcan_enki_`  
     
        `python3 cross_site_read_results.py --data_nm /ixi_camcan_enki_1000brains/4sites_ `


    9. **Cross-site: Get predictions from workflows**  
      `python3 cross_site_combine_predictions.py --model_folder /ixi_camcan_enki/ixi_camcan_enki_ --test_data_name /1000brains/1000brains_ --save_file_ext pred_1000brains_all`  
     
        `python3 cross_site_combine_predictions.py --model_folder /ixi_camcan_enki_1000brains/4sites_ --test_data_name /ADNI/ADNI_ --save_file_ext pred_adni_all`  
     
     
   10. **Cross site: Bias correction**  
    `python3 cross_site_bias_correction.py --data_path"     ../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains_S4_R4' --output_filenm ixi_camcan_enki_1000brains/4sites_S4_R4_pca_cv.gauss --mod_nm gauss`
