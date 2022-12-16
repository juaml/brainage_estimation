import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def bias_correction(train_data, test_data, x,  y):

    # bias correction using cole's method: (Using HC from the test sample)
    # a, b = np.polyfit(train_data[x], train_data[y], 1)
    # print(a, b)
    # corrected_predictions = (test_data[y] - b) / a
    # print(corrected_predictions)

    # bias correction using cole's method: (Using HC from the test sample)
    train_x = train_data[x].to_numpy().reshape(-1, 1)  # x = age
    train_y = train_data[y].to_numpy().reshape(-1, 1)  # y = predictions

    lin_reg = LinearRegression().fit(train_x, train_y)
    print(lin_reg.intercept_, lin_reg.coef_)

    corrected_predictions = (test_data[y] - lin_reg.intercept_[0]) / lin_reg.coef_[0][0]

    return corrected_predictions


if __name__ == '__main__':
    # Read arguments from submit file
    parser = argparse.ArgumentParser()
    parser.add_argument("--demographics_file", type=str, help="Demographics file path") # age and group is mandatory
    parser.add_argument("--predictions_file", type=str, help="Predictions file path")
    parser.add_argument("--predictions_column_name", type=str, help="Predictions", default='S4_R4_pca+gauss')
    parser.add_argument("--output_path", type=str, help="path to output_dir")  # eg'../results/ADNI'
    parser.add_argument("--output_prefix", type=str, help="prefix added to features filename ans results (predictions) file name", default='.BC') # eg: 'ADNI'

    # read arguments
    args = parser.parse_args()
    demographics_file = args.demographics_file
    predictions_file = args.predictions_file
    predictions_column_name = args.predictions_column_name
    output_path = (args.output_path)
    output_prefix = args.output_prefix


    # python3 cross_site_bias_correction_using_CN.py \
    #     --demographics_file ../data/ADNI/ADNI.subject_list_cat12.8.csv \
    #     --predictions_file ../results/ADNI/ADNI_S4_R4_pca.gauss_prediction.csv \
    #     --predictions_column_name S4_R4_pca+gauss \
    #     --output_path ../results/ADNI/ \
    #     --output_prefix _BC

    # python3 cross_site_bias_correction_using_CN.py \
    #     --demographics_file ../data/astronaut/demo_fake.csv \
    #     --predictions_file ../results/astronaut/filename_prediction.csv \
    #     --predictions_column_name S4_R4_pca+gauss \
    #     --output_path ../results/astronaut/ \
    #     --output_prefix _BC

    # demographics_file = '../data/ADNI/ADNI.subject_list_cat12.8.csv'
    # predictions_file = '../results/ADNI/ADNI_S4_R4_pca.gauss_prediction.csv'
    # predictions_column_name = 'S4_R4_pca+gauss'
    # output_path = '../results/ADNI/'
    # output_prefix = '_BC'

    # creating output filename same as imput predictions file name but with  prefix
    predictions_file_name_BC = predictions_file.replace('.csv', output_prefix + '.csv')

    demographics = pd.read_csv(demographics_file)
    predictions = pd.read_csv(predictions_file)

    # check if predictions contains predictions_column_name column as given by the user
    assert predictions_column_name in predictions.columns, f"{predictions_column_name} column not found in {predictions_file}"

    # check if demographics contains 'age' column and 'Research Group' column (which should have 'CN' as a category)
    assert "Research Group" in demographics.columns, f"'Research Group' column not found in {demographics_file}"
    assert "age" in demographics.columns, f"'age' column not found in {demographics_file}"
    assert 'CN' in demographics['Research Group'].unique(), f"'CN' group is not found in 'Research Group' column in {demographics_file}"

    # check if the demographics and predictions are of same length
    assert len(demographics) == len(predictions), "Mimatch between length of demographics and predictions"
    combined_df = pd.concat([demographics, predictions], axis=1)

    train_data = combined_df[combined_df["Research Group"] == "CN"]  # train only on Healthy subjects
    test_data = combined_df  # apply on whole sample
    x = 'age'
    y = predictions_column_name

    corrected_predictions = bias_correction(train_data=train_data, test_data=combined_df, x=x, y=y)
    corrected_predictions = corrected_predictions.to_frame()
    corrected_predictions = corrected_predictions.rename(columns={predictions_column_name: predictions_column_name + output_prefix}) # adding prefix to the column name

    corrected_predictions.to_csv(predictions_file_name_BC, index=False)



