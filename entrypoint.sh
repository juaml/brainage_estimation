#!/bin/bash
cd codes
python3 predict_age.py --features_path $1 --subject_filepaths $2 --output_path $3 --output_prefix $4 --mask_file ../masks/brainmask_12.8.nii --smooth_fwhm $5  --resample_size $6 --model_file $7



