import pandas
import numpy as np
from multiprocessing import Pool
import pickle
import os

from helper import train_val_split
from mimic_processing import fix_normalize_time

def map_column_to_ints(reddit_df, colname):
    list_values = reddit_df[colname].unique()
    num_values = len(list_values)
    map_values = {value: i for i, value in enumerate(list_values)}
    reddit_df[colname] = reddit_df[colname].map(map_values).astype(np.int32)
    assert(num_values == len(reddit_df[colname].unique()))
    return reddit_df

def read_csv(x):
    df = pandas.read_csv(x, header=None)
    df.columns = ['username', 'subreddit', 'utc']
    df['utc'] = pandas.to_datetime(df['utc'], unit='s')
    return df

def subreddit_data_to_df(data, time_threshold, colnames=None):
    base_path = './../data/dump/'
    files = os.listdir(base_path)
    file_list = [base_path+filename for filename in files if filename.startswith('partial_reddit_') and filename.split('.')[1]=='csv']
    print(files)

    with Pool(num_jobs) as pool:
        df_list = pool.map(read_csv, file_list)

    reddit_df = pandas.concat(df_list, ignore_index=True, sort=False)
    del df_list
    reddit_df = reddit_df.sort_values(by=['utc'], axis='rows')
    reddit_df = map_column_to_ints(reddit_df, 'subreddit')
    reddit_df = map_column_to_ints(reddit_df, 'username')

    # Remove users who have a duplicated timestamp
    users_to_remove_dupe = reddit_df[reddit_df.duplicated(subset=['username', 'utc'], keep=False)].username.unique()
    reddit_df = reddit_df[~reddit_df.username.isin(users_to_remove_dupe)]
    assert(len(reddit_df[reddit_df.duplicated(subset=['username', 'utc'])]['username'])==0)

    # Limit the users to only the ones with a certain frequency of posting
    freq_users = reddit_df.username.value_counts()
    hifreq_users = freq_users[freq_users>time_threshold].index.values
    print("Total users:",freq_users.shape)
    print("Users left after frequency pruning", hifreq_users.shape)
    del freq_users
    reddit_df = reddit_df[reddit_df.username.isin(hifreq_users)]

    reddit_df = reddit_df.reset_index(drop=True)
    return reddit_df


def extract_marker_from_group(user_idx):
    global group_users
    return group_users.get_group(user_idx)['subreddit'].values

def extract_time_from_group(user_idx):
    global group_users
    times = group_users.get_group(user_idx)['utc'].astype('int') / (10**9 * 3600 * 24)
    intervals = times - times.shift(periods=1)
    intervals.iloc[0] = 0
    return np.stack([times.values, intervals.values]).astype(np.int32).T

def get_markers_and_times(reddit_df):
    global group_users
    group_users = reddit_df.groupby('username')
    args = list(group_users.groups.keys())

    with Pool(num_jobs) as pool:
        x_data = pool.map(extract_marker_from_group, args)
        t_data = pool.map(extract_time_from_group, args)

    assert(np.all([t_data[idx].shape[1] == 2 for idx in range(100)]))
    assert(np.all([x_data[idx].shape[0] == t_data[idx].shape[0] for idx in range(100)]))
    assert np.all(group_users.subreddit.count().values == np.array(list(map(len, x_data))))
    assert np.all(group_users.utc.count().values == np.array(list(map(len, t_data))))

    return x_data, t_data

def save_subreddit_data(time_threshold, data=None):
    """
    Saves the data as a dict with keys 'x' and 't' into a file with path='path_data'
    The values of the keys are x_data and t_data:
    x_data: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
    t_data: list of length num_data_train, each element is numpy array of shape T_i x 2
    """

    reddit_df = subreddit_data_to_df(data, time_threshold)
    
    x_data, t_data = get_markers_and_times(reddit_df)
    t_data = fix_normalize_time(t_data)

    data_dict = {
        't': t_data,
        'x': x_data
    }
    #Train-Valid-Test Split of 60-20-20
    train_dict, extra_dict = train_val_split(data_dict, val_ratio=0.4)
    val_dict, test_dict = train_val_split(extra_dict, val_ratio=0.5)
    print(len(train_dict['x']), len(val_dict['x']), len(test_dict['x']))
    assert(len(train_dict['x']) == len(train_dict['t']))
    assert(len(val_dict['x']) == len(val_dict['t']))
    assert(len(test_dict['x']) == len(test_dict['t']))
    assert(len(train_dict['x']) + len(val_dict['x'])+ len(test_dict['x']) == len(data_dict['x']))

    train_path_data = '../data/subreddit_train.pkl'
    valid_path_data = '../data/subreddit_valid.pkl'
    test_path_data = '../data/subreddit_test.pkl'

    print("Saving training data to {}".format(train_path_data))
    with open(train_path_data, 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saving validation data to {}".format(valid_path_data))
    with open(valid_path_data, 'wb') as handle:
        pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saving testing data to {}".format(test_path_data))
    with open(test_path_data, 'wb') as handle:
        pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train_path_data, valid_path_data, test_path_data


if __name__ == "__main__":
    # only users with a series length less than time_threshold are chosen
    global num_jobs
    num_jobs = 4
    time_threshold = 100
    print(save_subreddit_data(time_threshold))