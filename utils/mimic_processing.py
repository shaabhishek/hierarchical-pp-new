import numpy as np
import os
import csv
import pickle

def valid_subject(path):
    #files = [os.path.join(path, o) for o in os.listdir(path) 
    #                if os.path.isdir(os.path.join(d,o)) is False]
    file = os.path.join(path, 'stays.csv')
    line_count = 0
    with open(file,mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            #print(row["ICUSTAY_ID"])
        #print(f'Processed {line_count} lines.')
    return line_count-1

def get_raw_data(path):
    stay_file = os.path.join(path, 'stays.csv')
    diag_file = os.path.join(path, 'diagnoses.csv')
    line_count = 0
    patient_dic = {}
    id_to_icu_map = {}
    fields = ["SUBJECT_ID","ICUSTAY_ID", "HADM_ID", "INTIME", "OUTTIME", "LOS", "ADMITTIME", "DISCHTIME","DEATHTIME", "ETHNICITY", "GENDER", "DOB", "AGE", "MORTALITY"]
    with open(stay_file,mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            episode_info = {}
            for f in fields:
                episode_info[f] = row[f]
            episode_info["ICD9_CODE"] = []
            id_to_icu_map[row["ICUSTAY_ID"]] = len(patient_dic)
            patient_dic[len(patient_dic)] = episode_info
    
    with open(diag_file,mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            id_ = id_to_icu_map[row["ICUSTAY_ID"]]
            patient_dic[id_]["ICD9_CODE"].append(row["ICD9_CODE"])
    return patient_dic



d = './../data/mimic3-benchmarks/data/root/'
subject_path = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
threshold = 3
valid_subject_list = [o for o in subject_path if valid_subject(o)>=threshold]
print(len(valid_subject_list), len(subject_path))

raw_data = [get_raw_data(o) for o in valid_subject_list]
#print(raw_data[0])
with open('./../data/mimic.pickle', 'wb') as handle:
    pickle.dump(raw_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

