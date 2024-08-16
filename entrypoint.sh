#!/bin/bash
cd codes
python3 predict_age_sing.py --features_path $1 --data_dir $2 --subject_filepaths $3 --output_path $4 --output_prefix $5 --mask_file ../masks/brainmask_12.8.nii --smooth_fwhm $6  --resample_size $7 --model_file $8
