import pickle
import pandas as pd

def read_data_cross_site(data_file, train_status, confounds):
    
    data_df = pickle.load(open(data_file, 'rb'))
    X = [col for col in data_df if col.startswith('f_')]
    y = 'age'
    data_df['age'] = data_df['age'].round().astype(int)  # round off age and convert to integer
    data_df = data_df[data_df['age'].between(18, 90)].reset_index(drop=True)
    duplicated_subs_1 = data_df[data_df.duplicated(['subject'], keep='first')] # check for duplicates (multiple sessions for one subject)
    data_df = data_df.drop(duplicated_subs_1.index).reset_index(drop=True)  # remove duplicated subjects

    if confounds is not None:  # convert sites in numbers to perform confound removal
        if train_status == 'train':
            site_name = data_df['site'].unique()
            if type(site_name[0]) == str:
                site_dict = {k: idx for idx, k in enumerate(site_name)}
                data_df['site'] = data_df['site'].replace(site_dict)

        elif train_status == 'test': # add site to features & convert site in a number to predict with model trained with  confound removal
            X.append(confounds)
            site_name = data_df['site'].unique()[0,]
            if type(site_name) == str:
                data_df['site'] = 10
    return data_df, X, y

