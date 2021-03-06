import numpy as np
import os
import csv
import pickle
import pandas
from sklearn.preprocessing import OneHotEncoder
from helper import train_val_split


def valid_subject(path):
    # files = [os.path.join(path, o) for o in os.listdir(path)
    #                if os.path.isdir(os.path.join(d,o)) is False]
    file = os.path.join(path, 'stays.csv')
    line_count = 0
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            # print(row["ICUSTAY_ID"])
        # print(f'Processed {line_count} lines.')
    return line_count - 1


def get_raw_data(path):
    stay_file = os.path.join(path, 'stays.csv')
    diag_file = os.path.join(path, 'diagnoses.csv')
    line_count = 0
    patient_dic = {}
    id_to_icu_map = {}
    fields = ["SUBJECT_ID", "ICUSTAY_ID", "HADM_ID", "INTIME", "OUTTIME", "LOS", "ADMITTIME",
              "DISCHTIME", "DEATHTIME", "ETHNICITY", "GENDER", "DOB", "AGE", "MORTALITY"]
    with open(stay_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            episode_info = {}
            for f in fields:
                episode_info[f] = row[f]
            episode_info["ICD9_CODE"] = []
            id_to_icu_map[row["ICUSTAY_ID"]] = len(patient_dic)
            patient_dic[len(patient_dic)] = episode_info

    with open(diag_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            id_ = id_to_icu_map[row["ICUSTAY_ID"]]
            patient_dic[id_]["ICD9_CODE"].append(row["ICD9_CODE"])
    return patient_dic


def preprocess_raw_data():
    d = './../data/mimic3-benchmarks/data/root/'
    subject_path = [os.path.join(d, o) for o in os.listdir(d)
                    if os.path.isdir(os.path.join(d, o))]
    threshold = 3
    valid_subject_list = [
        o for o in subject_path if valid_subject(o) >= threshold]
    print(len(valid_subject_list), len(subject_path))

    raw_data = [get_raw_data(o) for o in valid_subject_list]
    # print(raw_data[0])
    with open(raw_file, 'wb') as handle:
        pickle.dump(raw_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return raw_data


#############


def gender_to_categorical(x):
    return 0 if x == 'M' else 1


def mimic_correct_datatypes(icu_df):
    # TIMES
    icu_df['ADMITTIME'] = icu_df['ADMITTIME'].astype('datetime64')
    icu_df['DEATHTIME'] = icu_df['DEATHTIME'].astype('datetime64')
    icu_df['DISCHTIME'] = icu_df['DISCHTIME'].astype('datetime64')
    icu_df['INTIME'] = icu_df['INTIME'].astype('datetime64')
    icu_df['OUTTIME'] = icu_df['OUTTIME'].astype('datetime64')
    icu_df['DOB'] = icu_df['DOB'].astype('datetime64')

    # FLOATS
    icu_df['AGE'] = icu_df['AGE'].astype('float32')
    icu_df['LOS'] = icu_df['LOS'].astype('float32')

    # INTS
    icu_df['SUBJECT_ID'] = icu_df['SUBJECT_ID'].astype('int')
    icu_df['HADM_ID'] = icu_df['HADM_ID'].astype('int')
    icu_df['ICUSTAY_ID'] = icu_df['ICUSTAY_ID'].astype('int')
    icu_df['MORTALITY'] = icu_df['MORTALITY'].astype('int')
    return icu_df


def mimic_data_to_df(data, colnames=None):
    flat_data = [icu_visit for data_i in data for icu_visit in data_i.values()]
    icu_df = pandas.DataFrame(flat_data)
    icu_df = mimic_correct_datatypes(icu_df)
    # Make gender a 0/1 variable
    icu_df['GENDER'] = icu_df['GENDER'].apply(gender_to_categorical)

    # Make ETHNICITY a one-hot feature set, removing the first column to ensure there is no linear dependence
    #     ethnicity_dummies = pandas.get_dummies(icu_df.ETHNICITY, drop_first=True, prefix='ETHNICITY_')
    #     icu_df = pandas.concat([icu_df.drop('ETHNICITY', axis=1), ethnicity_dummies], axis=1)

    # Center the AGE column
    #     icu_df['AGE'] = (icu_df.AGE - icu_df.AGE.mean()) / icu_df.AGE.std()

    if colnames is None:
        return icu_df
    else:
        return icu_df[colnames]


def compute_markers(data):
    """
        Input: 
            data: list of dicts, where each dict corresponds to a patient and has 1 entry per icu visit
        Output:
            x_data: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
    """
    x_data = []
    icu_df = mimic_data_to_df(data)

    flat_icd9_codes = list(code for patient_codes in icu_df.ICD9_CODE for code in patient_codes)
    freqs_icd9 = pandas.value_counts(flat_icd9_codes)

    # Set the threshold frequency to be 150 to define 'high freq diseases'
    # Restricting the icd9 codes to those
    # which have occurrence frequency of more than threshold freq
    hifreq_icd9_codes = freqs_icd9[freqs_icd9 > 150].index
    set_icd9_codes = set(hifreq_icd9_codes)

    def get_primary_icd9(icd9_list):
        for x_i in icd9_list:
            if x_i in set_icd9_codes:
                return x_i
        return 'OTHER'

    icu_df['ICD9_CODE'] = icu_df.ICD9_CODE.apply(get_primary_icd9)

    icu_df = icu_df.reset_index(drop=True)

    # Make the disease as one-hot
    _onehotencoder = OneHotEncoder(sparse=False)
    _markers = _onehotencoder.fit_transform(icu_df.ICD9_CODE.values.reshape(-1, 1))

    group_patients = icu_df.groupby('SUBJECT_ID')
    for patient_idx, patient_df_rows in group_patients.groups.items():
        # Get an array of shape  (T_i x marker_dim)
        patient_markers = _markers[patient_df_rows].argmax(1)
        x_data.append(patient_markers)
    return x_data


def compute_times(data):
    """
        Input: 
            data: list of dicts, where each dict corresponds to a patient and has 1 entry per icu visit
        Output:
            t_data: list of length num_data_train, each element is numpy array of shape T_i x 3
    """
    #     max_visit_n = max([len(data_i) for data_i in data])
    t_data = []
    icu_df = mimic_data_to_df(data)

    group_patients = icu_df.groupby('SUBJECT_ID')
    for patient_idx, patient_df_rows in group_patients.groups.items():
        t_data_i = icu_df.loc[patient_df_rows][[
            'ADMITTIME', 'INTIME', 'OUTTIME']].astype('int') / (10 ** 9 * 3600 * 24)
        admit_times = t_data_i.ADMITTIME
        visit_durations = t_data_i.OUTTIME - t_data_i.INTIME
        # import pdb; pdb.set_trace()

        # Compute the intervals
        # Shift creates a NaT value at the first entry. Replace it with a zero in the end
        admit_intervals = admit_times - admit_times.shift(periods=1)
        admit_intervals.iloc[0] = 0

        # shifted = np.concatenate([np.zeros_like(admit_times[0:1]), admit_times[:-1]], axis=0)
        # intervals = admit_times - shifted
        # Convert nanoseconds to days
        # changing inervals at index 1 and visit duration at index 2
        t_data_i = np.stack([admit_times.values,
                             admit_intervals.values,
                             visit_durations.values]).T
        t_data.append(t_data_i)
    return t_data


def fix_normalize_time(data):
    N = len(data)
    admit_times = np.concatenate([i[:, 0] for i in data])
    interval_times = np.concatenate([i[1:, 1] for i in data])
    mean_admit_time, std_admit_time = np.mean(admit_times), np.std(admit_times)
    mean_interval_time, std_interval_time = np.median(
        interval_times), np.std(interval_times)

    for idx in range(N):
        data[idx][:, 0] = (data[idx][:, 0] - mean_admit_time) / std_admit_time
        data[idx][0, 1] = mean_interval_time
    return data


def save_mimic_data(data=None):
    """
    Saves the data as a dict with keys 'x' and 't' into a file with path='path_data'
    The values of the keys are x_data and t_data:
    x_data: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
    t_data: list of length num_data_train, each element is numpy array of shape T_i x 3
    """
    if data == None:
        raw_file = './../data/dump/mimic.pickle'
        with open(raw_file, 'rb') as handle:
            data = pickle.load(handle)

    x_data = compute_markers(data)
    t_data = compute_times(data)
    t_data = fix_normalize_time(t_data)

    assert (np.all([t_data[idx].shape[1] == 3 for idx in range(100)]))
    assert (np.all([x_data[idx].shape[0] == t_data[idx].shape[0]
                    for idx in range(100)]))

    data_dict = {
        't': t_data,
        'x': x_data
    }
    # Train-Valid-Test Split of 60-20-20
    train_dict, extra_dict = train_val_split(data_dict, val_ratio=0.4)
    val_dict, test_dict = train_val_split(extra_dict, val_ratio=0.5)
    assert (len(train_dict['x']) == len(train_dict['t']))
    assert (len(val_dict['x']) == len(val_dict['t']))
    assert (len(test_dict['x']) == len(test_dict['t']))
    assert (len(train_dict['x']) + len(val_dict['x']) +
            len(test_dict['x']) == len(data_dict['x']))

    train_path_data = '../data/mimic_train.pkl'
    valid_path_data = '../data/mimic_valid.pkl'
    test_path_data = '../data/mimic_test.pkl'

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
    # raw_file = './../data/dump/mimic.pickle'
    # if os.path.isfile(raw_file):
    #     print("Dictionary Exists!")
    #     with open(raw_file, 'rb') as handle:
    #         data = pickle.load(handle)
    # else:
    #     data = preprocess_raw_data()
    # print(len(data))
    print(save_mimic_data())
    # raw_file = './../data/dump/mimic.pickle'
    # with open(raw_file, 'rb') as handle:
    #     data = pickle.load(handle)
    # values = [v['ICD9_CODE'] for data_i in data for v in data_i.values()] 
    # count = {}
    # for v in values:
    #     for j in v:
    #         if j not in count:
    #             count[j]=1
    #         else:
    #             count[j] += 1
    # ls = set([k for k, v in  count.items() if v>150])
    # print(len(ls))

    # f_count =0
    # o_count =0
    # for v in values:
    #     if v[0] in ls:
    #         f_count +=1
    #         continue
    #     else:
    #         for j in v:
    #             if j in ls:
    #                 o_count +=1
    #                 break
    # print(f_count, o_count, len(values)- (f_count+o_count))
