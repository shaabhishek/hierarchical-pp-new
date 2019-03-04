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



filename = './../data/dump/stack_exchange.pickle'
if os.path.isfile(filename):
    with open(filename, 'rb') as handle:
        print("Dictionary Exists!")
        data = pickle.load(handle)
        print(len(data))
else:
    data = get_elements()
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
