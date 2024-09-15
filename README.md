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

Alternatively, you may use Mamba to create an environment including all of the above:
```
$ git clone https://github.com/juaml/brainage_estimation.git
$ cd brainage_estimation
$ mamba env create --file requirements.yml
$ mamba activate brainage_estimation
```

After the set up following codes can be run as provided in the `codes` directory.

2. **Get predictions** 

We provide pretrained models that can used to obtain predictions on new samples.

```
python3 predict_age.py \
    --features_path path_to_features_dir \
    --subject_filepaths path_to_txt_file \            
    --output_path path_to_output_dir \            
    --output_prefix PREFIX \         
    --mask_file ../masks/brainmask_12.8.nii \            
    --smooth_fwhm 4 \
    --resample_size 4 \
    --model_file ../trained_models/4sites.S4_R4_pca.gauss.models
```

The arguments are:
- `--features_path` should point to a directory where calculated features are stored as a `pickle` file.
- `--subject_filepaths` should point to a text file containing path to the CAT12.8's `mwp1` file for each subject per line.
- `--output_path` points to a directory where the predictions will be saved.
- `--output_prefix` prefix for the output files.
- `--mask_file` points to the GM mask to be used (defaults to `../masks/brainmask_12.8.nii`)
- `--smooth_fwhm` smoothing kernel size to be used (defaults to `4`)
- `--resample_size` resampling of the voxels to isometric size (defaults to `4`)
- `--model_file` should point to an already trained model (defaults to `4sites_S4_R4_pca.gauss.models`)
             
This will calculate features with 4mm smoothing and 4mm resampling (`S4_R4`) for all subjects in the file provided via `--subject_filepaths`.
The predictions will be performed using the S4_R4_pca+gauss model.
The model will perform `PCA` based on the model used.
Note that if the features are available in the `--features_path` then they will not be recalculated.

3. **calculate features: voxel-wise and parcel-wise features**
        
It is possible to calculate features from a list of CAT12.8 files.

Voxel-wise features
```
python3 calculate_features_voxelwise.py \
    --features_path ../data/ADNI/ \
    --subject_filepaths ../data/ADNI/ADNI.paths_cat12.8.csv \
    --output_prefix ADNI \
    --mask_file ../masks/brainmask_12.8.nii \
    --smooth_fwhm 4 \
    --resample_size 8 \
```

Parcel-wise features
```
python3 calculate_features_parcelwise.py \
    --features_path ../data/ADNI/ \
    --subject_filepaths ../data/ADNI/ADNI.paths_cat12.8.csv \
    --output_prefix ADNI \
    --mask_file ../masks/BSF_173.nii \
    --num_parcels 173 \
```
    
4. **Within-site: Train models**
        
```
python3 within_site_train.py \
    --demographics_file ../data/ixi/ixi.subject_list_cat12.8.csv \
    --features_file ../data/ixi/ixi.173 \
    --output_path ../results/ixi \
    --output_prefix ixi.173 \
    --models rvr_lin \
    --pca_status 0
```

The arguments are:
- `--demographics_file` should point to a `csv` file with four columns `{'subject', 'site', 'age', 'gender'}`.
- `--features_file` should point to a `pickle` file with features.
- `--output_path` points to a directory where the models, scores and results will be saved.
- `--output_prefix` prefix for output files which will be used to create three files `.models`, `.scores`, and `.results`.
- `--models` one or more models to train, multiple models can be provided as a comma separated list.
- `--pca_status` either 0 (no PCA) or 1 (for PCA retaining 100% variance). 

This will run outer 5-fold and inner 5x5-fold cross-validation.

In case you are using `HTcondor`, you can also use the provided submit file.

`condor_submit within_site_ixi.submit`


5. **Within-site: Read results from saved models**  
        
`python3 within_site_read_results.py --data_nm ../results/ixi/ixi.`


6. **Within-site: Get predictions from 128 workflows**  
        
```
python3 within_site_combine_predictions.py \
    --demographics_file ../data/ixi/ixi.subject_list_cat12.8.csv \
    --features_path ../data/ixi/ixi. \
    --model_path ../results/ixi/ixi. \
    --output_prefix all_models_pred
 ```
        
7. **Within-site: Bias correction**
        
```
python3 within_site_bias_correction.py \
    --input_predictions_file ../results/ixi/ixi.all_models_pred.csv \
    --BC_predictions_file ../results/ixi/ixi.all_models_pred_BC.csv
```


8. **Cross-site: Train and test**  
      
First train a model with three sites.
```
python3 cross_site_train.py \
    --demographics_file ../data/ixi_camcan_enki/ixi_camcan_enki_subject_list_cat12.8.csv \
    --features_file ../data/ixi_camcan_enki/ixi_camcan_enki.173 \
    --output_path ../results/ixi_camcan_enki \
    --output_prefix ixi_camcan_enki.173 \
    --models rvr_lin \
    --pca_status 0
```

Now we can make predictions on the hold-out site using all models available in the `--model_path`.
```  
python3 cross_site_combine_predictions.py \
    --demographics_file ../data/1000brains/1000brains.subject_list_cat12.8.csv \
    --features_path ../data/1000brains/1000brains. \
    --model_path ../results/ixi_camcan_enki/ixi_camcan_enki. \
    --output_prefix pred_1000brains_all

```

9. **Cross-site: Read results from saved models**  
        
Create cross-validation scores from cross-site predictions.
        
`python3 cross_site_read_results.py --data_nm ../results/ixi_camcan_enki/ixi_camcan_enki.`

     
10. **Cross-site: Bias correction**

Using the CV predictions from the training data:

```
python3 cross_site_bias_correction.py \
    --demographics_file ../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains.subject_list_cat12.8.csv \
    --features_file ../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains.S4_R4 \
    --model_file ../results/ixi_camcan_enki_1000brains/4sites.S4_R4_pca_cv.gauss
```

Using the control subjects from the testing data: 

This code will train bias correction model using the predictions and age from the control group (`CN`) group and apply to it the full sample. It needs `demographics_file` which should contain `age` and `Research Group` columns, and `Research Group` column should contain `CN` category. `predictions_file` should contain a column for predictions defined by `predictions_column_name`. The bias corrected predictions will be saved in the same location as `predictions_file` with a prefix defined by `output_prefix`.

```
python3 cross_site_bias_correction_using_CN.py \
    --demographics_file ../data/ADNI/ADNI.subject_list_cat12.8.csv \
    --predictions_file ../results/ADNI/ADNI.S4_R4_pca.gauss.prediction.csv \
    --predictions_column_name S4_R4_pca+gauss \
    --output_prefix _BC
```
