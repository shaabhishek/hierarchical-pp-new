import pickle
import numpy as np
import pandas
import xml.etree.ElementTree as ET
import datetime
import os

from helper import train_val_split
from mimic_processing import fix_normalize_time

def is_date_between(time_ , start_date= "2012-01-01", end_date = "2014-01-02"):
    start_date = start_date + "T00:00:00.000"
    end_date = end_date + "T00:00:00.000"
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S.%f')
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S.%f')
    time_obj = datetime.datetime.strptime(time_, '%Y-%m-%dT%H:%M:%S.%f')
    if start_date_obj<=time_obj and time_obj<end_date_obj:
        return True
    return False

def get_elements(path ='./../data/Badges.xml'):
    root = ET.parse(path).getroot()
    dics = []
    count = 0
    fields = ["Id", "UserId", "Name", "Date"]
    for child in root:
        count += 1
        if count%10000 == 0:
            print("Count is :",count)
        z =child.attrib
        if is_date_between(z['Date']):
            dics.append([z[f] for f in fields])
    return dics

def process_elements(data):
    dics = {}

    for item in data:
        userid, name, date = item[1], item[2], item[3]
        if userid not in dics:
            dics[userid] = []
        if name[0].isupper():
            dics[userid].append([name, date])

    name_stat = {}
    new_dics ={}
    student_stat = []
    for userid in dics:
        items = dics[userid]
        student_stat.append(len(items))
        if len(items) <40:
            continue
        new_dics[userid] = dics[userid]
        names = {}
        for i in items:
            tag = i[0]
            if tag not in names:
                names[tag] = 1.
            else:
                names[tag] = 1. + names[tag]
        for tags in names:
            if tags not in name_stat:
                name_stat[tags] = []
            name_stat[tags].append(names[tags])

    print("Number of student\t Mean \t Std \t Min\t Max\t new User")
    student_stat = np.asarray(student_stat)
    print(student_stat.size, np.mean(student_stat), np.std(student_stat), np.min(student_stat), np.max(student_stat), len(new_dics) )
    #Print stats
    print("tags\t Total \t Min \t Max\t Mean \t Std\t Unique user")
    remove_tags = {}
    for tags in name_stat:
        vals = np.asarray(name_stat[tags])
        print(tags+"\t" + str(np.sum(vals)) +"\t"+ str(np.min(vals)) +"\t"+ str(np.max(vals)) +"\t"+ str(np.mean(vals))+"\t" +str(np.std(vals)) +"\t"+str(vals.shape[0])                    )
        if np.mean(vals)< 1.01:
            remove_tags[tags] = 1.

    final_dics = {}
    for userid in new_dics:
        items = new_dics[userid]
        new_items = []
        for i in items:
            if i[0] not in remove_tags:
                new_items.append(i)
        final_dics[userid] = new_items
    return final_dics



# filename = './../../data/dump/stack_exchange.pickle'
# if os.path.isfile(filename):
#     with open(filename, 'rb') as handle:
#         print("Raw Dictionary Exists!")
#         data = pickle.load(handle)
#         print(len(data))
# else:
#     data = get_elements()
#     with open(filename, 'wb') as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# filename = './../../data/dump/stack_processed.pickle'
# if os.path.isfile(filename):
#     with open(filename, 'rb') as handle:
#         print("Processed Dictionary Exists!")
#         data = pickle.load(handle)
#         print(len(data))
# else:
#     data = process_elements(data)
#     with open(filename, 'wb') as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def stackex_correct_datatypes(stackex_df):

    # TIMES
    stackex_df['timestamp'] = stackex_df['timestamp'].astype('datetime64[ms]')

    # INTS
    stackex_df['userid'] = stackex_df['userid'].astype('int')
    return stackex_df

def stackex_data_to_df(data, colnames=None):
    flat_data = [{'userid':idx_user, 'marker': x, 'timestamp': t} for idx_user, data_user in data.items() for x,t in data_user]
    stackex_df = pandas.DataFrame(flat_data)
    stackex_df = stackex_correct_datatypes(stackex_df)
    
    # Make marker a categorical variable
    unique_marker_labels = stackex_df['marker'].unique()
    stackex_df['marker'] = stackex_df['marker'].apply(lambda x: np.argwhere(x==unique_marker_labels).item())
    
    # Remove userid's with duplicate timestamps as per the RMTPP paper
    # Also remove userid's with length > 200
    group_users = stackex_df.groupby('userid')
    stackex_df = group_users.filter(lambda x: (len(x)<200) and ~np.any(x.duplicated('timestamp')))
    
#     Sort values first by userid and then by timestamp..
    stackex_df = stackex_df.sort_values(by=['userid', 'timestamp', 'marker'], axis='rows')
    
    return stackex_df


def compute_markers(data, stackex_df=None):
    """
        Input: 
            data: dict of lists, where each key is a user and each value is a list of [x, t] entries for the user
        Output:
            x_data: list of length num_data_train, each element is numpy array of shape T_i x 1
    """
    x_data = []
    if stackex_df is None:
        stackex_df = stackex_data_to_df(data)
    group_users = stackex_df.groupby('userid')

    for user_idx, user_df_rows in group_users.groups.items():
        user_markers = stackex_df.loc[user_df_rows]['marker']
        user_markers = user_markers.values.reshape(len(user_df_rows),-1)
    #     t_i, marker_dim = user_markers.shape # marker_dim will be 1 here
        x_data.append(user_markers)
    return x_data

def compute_times(data, stackex_df=None):
    """
        Input: 
            data: dict of lists, where each key is a user and each value is a list of [x, t] entries for the user
        Output:
            t_data: list of length num_data_train, each element is numpy array of shape T_i x 2
    """
    t_data = []
    if stackex_df is None:
        stackex_df = stackex_data_to_df(data)
    group_users = stackex_df.groupby('userid')

    for user_idx, user_df_rows in group_users.groups.items():
        user_times = stackex_df.loc[user_df_rows]['timestamp']
        user_times = user_times.astype('int') / (10**9 * 3600 * 24 * 365.25)
        # Shift creates a NaT value at the first entry. Replace it with a zero in the end
        user_intervals = user_times - user_times.shift(periods=1)
        user_intervals.iloc[0] = 0
        t_data_i = np.stack([user_times.values, user_intervals.values], axis=1)
        t_data.append(t_data_i)
    
    return t_data


def save_stackex_data(data=None):
    """
    Saves the data as a dict with keys 'x' and 't' into a file with path='path_data'
    The values of the keys are x_data and t_data:
    x_data: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
    t_data: list of length num_data_train, each element is numpy array of shape T_i x 3
    """
    if data==None:
        filename = './../data/dump/stack_processed.pickle'
        try: 
            # if os.path.isfile(filename):
            with open(filename, 'rb') as handle:
                print("Processed Dictionary Exists!")
                data = pickle.load(handle)
                print('Data length: ', len(data))
        except:
            print('Data didnt load')

    stackex_df = stackex_data_to_df(data)
    assert(len(stackex_df[stackex_df.duplicated(subset=['userid', 'timestamp'])]['userid'])==0)
    t_data = compute_times(data, stackex_df)
    t_data = fix_normalize_time(t_data)
    x_data = compute_markers(data, stackex_df)
    assert(np.all([t_data[idx].shape[1] == 2 for idx in range(100)]))
    assert(np.all([x_data[idx].shape[0] == t_data[idx].shape[0] for idx in range(100)]))

    data_dict = {
        't': t_data,
        'x': x_data
    }
    #Train-Valid-Test Split of 60-20-20
    train_dict, extra_dict = train_val_split(data_dict, val_ratio=0.4)
    val_dict, test_dict = train_val_split(extra_dict, val_ratio=0.5)
    assert(len(train_dict['x']) == len(train_dict['t']))
    assert(len(val_dict['x']) == len(val_dict['t']))
    assert(len(test_dict['x']) == len(test_dict['t']))
    assert(len(train_dict['x']) + len(val_dict['x'])+ len(test_dict['x']) == len(data_dict['x']))

    train_path_data = '../data/stackexchange_train.pkl'
    valid_path_data = '../data/stackexchange_valid.pkl'
    test_path_data = '../data/stackexchange_test.pkl'

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
    print(save_stackex_data())






