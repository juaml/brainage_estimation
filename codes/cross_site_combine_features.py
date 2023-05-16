import pickle
import pandas as pd
import os.path

if __name__ == '__main__':

    results_folder = '../data/ixi_camcan_enki_1000brains/ixi_camcan_enki_1000brains.'
    data_list = ['../data/ixi/ixi.', '../data/camcan/camcan.', '../data/enki/enki.', '../data/1000brains/1000brains.']

    results_folder = '../data/ixi_camcan_enki/ixi_camcan_enki.'
    data_list = ['../data/ixi/ixi.', '../data/camcan/camcan.', '../data/enki/enki.']

    results_folder = '../data/ixi_camcan_1000brains/ixi_camcan_1000brains.'
    data_list = ['../data/ixi/ixi.', '../data/camcan/camcan.', '../data/1000brains/1000brains.']
    
    results_folder = '../data/camcan_enki_1000brains/camcan_enki_1000brains_'
    data_list = ['../data/camcan/camcan.', '../data/enki/enki.', '../data/1000brains/1000brains.']

    results_folder = '../data/ixi_enki_1000brains/ixi_enki_1000brains.'
    data_list = ['../data/ixi/ixi.', '../data/enki/enki.', '../data/1000brains/1000brains.']

    feature_list = ['173', '473', '873', '1273', 'S0_R4', 'S4_R4', 'S8_R4', 'S0_R8', 'S4_R8', 'S8_R8']

    combined_data_df = pd.DataFrame()
    combined_demo_df = pd.DataFrame()

    for feature_item in feature_list:
        print(feature_item)

        combined_data_df = pd.DataFrame()
        combined_demo_df = pd.DataFrame()

        for data_item in data_list:
            datafile_name = data_item + feature_item + '.csv'
            demofile_name = data_item + 'subject_list_cat12.8.csv'
            print(datafile_name, demofile_name)

            if os.path.exists(datafile_name):

                data_df, demo_df = pd.read_csv(datafile_name), pd.read_csv(demofile_name)
                print(data_df.shape, demo_df.shape)

                if 'session' not in demo_df.columns:
                    demo_df['session'] = 'ses-1'

                combined_data_df = pd.concat([combined_data_df, data_df])
                combined_demo_df = pd.concat([combined_demo_df, demo_df])
            else:
                break

        combined_data_df = combined_data_df.reset_index(drop=True)
        combined_demo_df = combined_demo_df.reset_index(drop=True)

        print(combined_data_df.shape, combined_demo_df.shape)

#        demographic_file = results_folder + 'subject_list_cat12.8.csv'
#        features_file = results_folder + feature_item
#        print(demographic_file, features_file)
#
#        combined_demo_df.to_csv(demographic_file, index=False)
#        combined_data_df.to_csv(features_file + '.csv', index=False)
#        pickle.dump(combined_data_df, open(features_file, 'wb'), protocol=4)






