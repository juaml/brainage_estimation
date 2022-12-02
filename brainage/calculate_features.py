import os.path
import nilearn
from nilearn import image
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.processing as npr

def subsample_img(img, f):
    """Reduce resample_to_img features of a 3D array by a given factor f."""

    data = img.get_fdata()
    mask = np.zeros(img.shape)
    mask[::f, ::f, ::f] = 1
    data = data * mask
    return nib.Nifti1Image(data, img.affine, img.header)

def binarize_3d(img, threshold):
    """binarize 3D spatial image"""
    return nib.Nifti1Image(
        np.where(img.get_fdata() > threshold, 1, 0), img.affine, img.header
    )

def calculate_voxelwise_features(phenotype_file, mask_file, smooth_fwhm, resample_size):
    """Calculate voxelwise features for the subjects

    Args:
        phenotype_file (csv or txt): A csv or text file with path to subject images
        mask_file (nii): The GM mask file to be used to extract features
        smooth_fwhm (int): Smooth images by applying a Gaussian filter by given FWHM (mm)
        resample_size (int): Resample image to given voxel size

    Returns:
        data_resampled (dataframe): pandas dataframe of features (N subjects by M features)
    """    

    phenotype = pd.read_csv(phenotype_file, header=None)
    
    # don't need this anymore
    # filename, file_extension = os.path.splitext(phenotype_file)
    # if file_extension == ".txt":
    #     phenotype = pd.read_csv(phenotype_file, header=None)
    # elif file_extension == ".csv":
    #     phenotype = pd.read_csv(phenotype_file, sep=",", header=None)
    # else:
    #     raise ValueError("Wrong file. Please imput either a csv or text file")

    print(phenotype.shape)
    print(phenotype.head())

#    phenotype = phenotype.iloc[0:15]

    data_resampled = np.array([])  # array to save resampled features from subjects mri
    count = 0
    for index, row in phenotype.iterrows():  # iterate over each row
        sub_file = row.values[0]

        if os.path.exists(sub_file):
            print(f"\n-----Processing subject number {count}------")
            sub_img = nib.load(sub_file)  # load subject image
            mask_img = nib.load(mask_file)  # load mask image
            print("Subject and mask image loaded")
            print("sub affine original \n", sub_img.affine, sub_img.shape)
            print("mask affine original \n", mask_img.affine, mask_img.shape)

            print("Perform smoothing")
            sub_img = image.smooth_img(
                sub_img, smooth_fwhm
            )  # smooth the image with 4 mm FWHM

            print("Perform resampling")
            # trying to match Gaser
            mask_img_rs = npr.resample_to_output(
                mask_img, [resample_size] * len(mask_img.shape), order=1
            )  # resample mask
            print(
                "mask affine after resampling\n",
                mask_img_rs.affine,
                mask_img_rs.shape,
            )

            sub_img_rs = image.resample_to_img(
                sub_img, mask_img_rs, interpolation="linear"
            )  # resample subject
            print(
                "sub affine after resampling\n",
                sub_img_rs.affine,
                sub_img_rs.shape,
            )

            binary_mask_img_rs = binarize_3d(mask_img_rs, 0.5)  # binarize the mask
            mask_rs = binary_mask_img_rs.get_fdata().astype(bool)

            sub_data_rs = sub_img_rs.get_fdata()[
                mask_rs
            ]  # extract voxel using the binarized mask
            sub_data_rs = sub_data_rs.reshape(1, -1)

            if data_resampled.size == 0:
                data_resampled = sub_data_rs
            else:
                data_resampled = np.concatenate((data_resampled, sub_data_rs), axis=0)
            count = count + 1
            print(data_resampled.shape)

    print("\n *** Feature extraction done ***")

    # renaming the columns and convering to dataframe
    data_resampled = pd.DataFrame(data_resampled)
    data_resampled.rename(columns=lambda X: "f_" + str(X), inplace=True)
    print('Feature names:', data_resampled.columns)

    print(f"The size of the feature space is {data_resampled.shape}")

    return data_resampled



def calculate_parcelwise_features(phenotype_file, mask_dir, num_parcels):
    """Calculate parcelwise features for the subjects

    Args:
        phenotype_file (csv or text): A csv or text file with path to subject images
        mask_dir (_type_): The GM mask file to be used to extract features
        num_parcels (_type_): Number of parcels
    
    Returns:
        data_parcels (dataframe): pandas dataframe of features (N subjects by M parcels)
    """    

    phenotype = pd.read_csv(phenotype_file, header=None)

    # filename, file_extension = os.path.splitext(phenotype_file)

    # if file_extension == '.txt':
    #     phenotype = pd.read_csv(phenotype_file, header=None)
    # elif file_extension == '.csv':
    #     phenotype = pd.read_csv(phenotype_file, sep=',', header=None)
    # else:
    #     raise ValueError("Wrong file. Please imput either a csv or text file")

    print(phenotype.shape)
    print(phenotype.head())
#    phenotype = phenotype.iloc[0:15]

    data_parcels = [] #np.array([])  # array to save resampled features from subjects mri
    count = 0

    for index, row in phenotype.iterrows(): # iterate over each row
        sub_file = row.values[0]

        if os.path.exists(sub_file):
            print(f'\nProcessing subject number {count}')
            sub_img = nib.load(sub_file)  # load subject image
            mask_img = nib.load(mask_dir)  # load mask image
            print ('Subject and mask image loaded')
            print(sub_file, sub_img.affine, mask_img.affine)

            sub_data = sub_img.get_fdata()
            sub_data[sub_data == 0] = np.nan # replace zeros with Nan
            sub_data_parcels = []

            if not np.array_equal(sub_img.affine, mask_img.affine):
                mask_img = nilearn.image.resample_to_img(mask_img, sub_img, interpolation='linear')
            else:
                print("Subject and mask have same affine")

            for num in range(1, int(num_parcels) + 1):
                itemindex = np.where(mask_img.get_fdata() == num)  # get indices from the mask for a parcel
                sub_mat = sub_data[itemindex]

                if np.all(np.isnan(sub_mat)):
                    sub_agg = 0
                else:
                    sub_agg = np.nanmean(sub_mat) # mean the data from the indices to get GM volume
                sub_data_parcels.append(sub_agg)

            data_parcels.append(sub_data_parcels)
            print(len(data_parcels))
            count = count + 1

    print('\n *** Feature extraction done ***')
    data_parcels = pd.DataFrame(data_parcels)
    data_parcels.rename(columns=lambda X :'f_' + str(X), inplace=True)
    print(data_parcels.columns)

    print('final dataframe shape', data_parcels.shape)
    return data_parcels
