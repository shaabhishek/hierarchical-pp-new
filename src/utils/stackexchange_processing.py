import pickle
import numpy as np
import xml.etree.ElementTree as ET
import datetime
import os


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



filename = './../../data/dump/stack_exchange.pickle'
if os.path.isfile(filename):
    with open(filename, 'rb') as handle:
        print("Raw Dictionary Exists!")
        data = pickle.load(handle)
        print(len(data))
else:
    data = get_elements()
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

filename = './../../data/dump/stack_processed.pickle'
if os.path.isfile(filename):
    with open(filename, 'rb') as handle:
        print("Processed Dictionary Exists!")
        data = pickle.load(handle)
        print(len(data))
else:
    data = process_elements(data)
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)








