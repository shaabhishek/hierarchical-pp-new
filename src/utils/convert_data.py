import numpy as np
import pandas
import pickle
import random
from helper import train_val_split

def seq_to_marker(seq):
    """
        Input: dict with keys: 'time_since_start', 'time_since_last_event'
    """
    seq = sorted(seq, key=lambda x: x['time_since_start'])
    markers = np.array([datum['type_event'] for datum in seq])
    return markers

def seq_to_times(seq):
    """
        Input: dict with keys: 'time_since_start', 'time_since_last_event'
    """
    seq = sorted(seq, key=lambda x: x['time_since_start'])
    times = np.array([(datum['time_since_last_event'], datum['time_since_start']) for datum in seq])
    return times

def getdata_NHP(filepath, max_size=None):
    t_data, x_data = [],[]
    with open(filepath,'rb') as f:
        data = pickle.load(f, encoding='latin1')
        for _data in data.values():
            if (type(_data)==list) and len(_data)>0:
                x_data.extend(list(map(seq_to_marker, _data)))
                t_data.extend(list(map(seq_to_times, _data)))
    if max_size is not None:
        idxs = random.sample(range(len(x_data)), max_size)
        t_data = [t_data[idx] for idx in idxs]
        x_data = [x_data[idx] for idx in idxs]

    data_dict = {
        't': t_data,
        'x': x_data
    }
    return data_dict

def list_to_stacked_time_array(x):
    intervals = pandas.Series(x) - pandas.Series(x).shift(1)
    try:
        intervals[0] = x[0]
    except:
        import pdb; pdb.set_trace()
    return np.stack([intervals, pandas.Series(x)]).T

def getdata_Hawkes(filepath):
    t_data, x_data = [],[]
    with open(filepath,'r') as f:
        data=f.readlines()

    print(len(data))
    # Split a line into list of floats
    random.shuffle(data)
    data = list(map(lambda x: list(map(float, str.split(x))), data))
    t_data = list(map(list_to_stacked_time_array, data))
    x_data = list(map(lambda x: np.ones(len(x)), data))
    data_dict = {
        't': t_data,
        'x': x_data
    }
    return data_dict

def convert_dataset(data_name):
    if data_name not in {'lastfm', 'meme', 'retweet', 'syntheticdata_nclusters_5', 'syntheticdata_nclusters_10', 'syntheticdata_nclusters_50', 'syntheticdata_nclusters_100'}:
        for idx in range(1,6):
            for ls in ['train', 'test']:
                event_data_path = './../data/real/'+data_name+'/'+ 'event-'+str(idx)+'-'+ls+'.txt'
                time_data_path = './../data/real/'+data_name+'/'+ 'time-'+str(idx)+'-'+ls+'.txt'
                events = []
                times = []
                with open(event_data_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        values = np.asarray([int(i) for i in line.strip().split(' ')])-1
                        events.append(values)

                with open(time_data_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        x = np.asarray([float(i) for i in line.strip().split(' ')])
                        z = x[1:] - x[:-1]
                        y = np.zeros(x.shape)
                        y[1:] = z
                        values = np.concatenate([y.reshape(-1,1),x.reshape(-1,1)], axis = 1)
                        if data_name =='so':
                            values[:,1 ] = values[:,1]-1325385742.
                            times.append(values/(24.*3600))#0.00001)
                        if data_name == 'book_order':
                            values[:,0 ] = values[:,0]*100.
                            times.append(values*1)
                        else:
                           times.append(values)

                assert len(events)== len(times), "Len mismatch"
                print(ls, idx, len(times))
                data_dict = {
                't': times,
                'x': events
                }
                if ls == 'train':
                    train_dict, val_dict = train_val_split(data_dict, val_ratio=0.1)

                    train_file = '../data/'+data_name+'_'+str(idx)+'_train.pkl'
                    valid_file = '../data/'+data_name+'_'+str(idx)+'_valid.pkl'
                    with open(train_file, 'wb') as handle:
                        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(valid_file, 'wb') as handle:
                        pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:##Test Dataset
                    test_file = '../data/'+data_name+'_'+str(idx)+'_test.pkl'
                    with open(test_file, 'wb') as handle:
                        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if data_name =='lastfm':
        for ls in ['train', 'test']:
                event_data_path = './../data/real/'+data_name+'/'+ 'event_split_1000'+'-'+ls+'.txt'#event_split_1000-test.txt
                time_data_path = './../data/real/'+data_name+'/'+ 'time_split_1000'+'-'+ls+'.txt'
                events = []
                times = []
                with open(event_data_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        values = np.asarray([int(i) for i in line.strip().split(' ')])-1
                        events.append(values)

                with open(time_data_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        x = np.asarray([float(i) for i in line.strip().split(' ')])
                        z = x[1:] - x[:-1]
                        y = np.zeros(x.shape)
                        y[1:] = z
                        values = np.concatenate([y.reshape(-1,1),x.reshape(-1,1)], axis = 1)
                        times.append(values)
                assert len(events)== len(times), "Len mismatch"
                print(ls, len(times))
                data_dict = {
                't': times,
                'x': events
                }
                if ls == 'train':
                    train_dict, val_dict = train_val_split(data_dict, val_ratio=0.1)

                    train_file = '../data/'+data_name+'_'+str(1)+'_train.pkl'
                    valid_file = '../data/'+data_name+'_'+str(1)+'_valid.pkl'
                    with open(train_file, 'wb') as handle:
                        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(valid_file, 'wb') as handle:
                        pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:##Test Dataset
                    test_file = '../data/'+data_name+'_'+str(1)+'_test.pkl'
                    with open(test_file, 'wb') as handle:
                        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if data_name == 'meme':
            valid_dict = getdata_NHP('./../data/dump/data_meme/dev.pkl')
            train_dict = getdata_NHP('./../data/dump/data_meme/train.pkl', 32000)
            test_dict = getdata_NHP('./../data/dump/data_meme/test.pkl')

            train_file = '../data/'+data_name+'_'+str(1)+'_train.pkl'
            valid_file = '../data/'+data_name+'_'+str(1)+'_valid.pkl'
            test_file = '../data/'+data_name+'_'+str(1)+'_test.pkl'
            
            with open(train_file, 'wb') as handle:
                pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(valid_file, 'wb') as handle:
                pickle.dump(valid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(test_file, 'wb') as handle:
                pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if data_name == 'retweet':
            valid_dict = getdata_NHP('./../data/dump/data_retweet/dev.pkl')
            train_dict = getdata_NHP('./../data/dump/data_retweet/train.pkl')
            test_dict = getdata_NHP('./../data/dump/data_retweet/test.pkl')

            train_file = '../data/'+data_name+'_'+str(1)+'_train.pkl'
            valid_file = '../data/'+data_name+'_'+str(1)+'_valid.pkl'
            test_file = '../data/'+data_name+'_'+str(1)+'_test.pkl'
            
            with open(train_file, 'wb') as handle:
                pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(valid_file, 'wb') as handle:
                pickle.dump(valid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(test_file, 'wb') as handle:
                pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if data_name in ['syntheticdata_nclusters_5', 'syntheticdata_nclusters_10', 'syntheticdata_nclusters_50', 'syntheticdata_nclusters_100']:
            train_dict = getdata_Hawkes('./../data/dump/{}_train.txt'.format(data_name))
            valid_dict = getdata_Hawkes('./../data/dump/{}_val.txt'.format(data_name))
            test_dict = getdata_Hawkes('./../data/dump/{}_test.txt'.format(data_name))

            train_file = '../data/'+data_name+'_'+str(1)+'_train.pkl'
            valid_file = '../data/'+data_name+'_'+str(1)+'_valid.pkl'
            test_file = '../data/'+data_name+'_'+str(1)+'_test.pkl'
            
            with open(train_file, 'wb') as handle:
                pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(valid_file, 'wb') as handle:
                pickle.dump(valid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(test_file, 'wb') as handle:
                pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    #convert_dataset('mimic2')
    # convert_dataset('so')
    #convert_dataset('meme')
    # convert_dataset('lastfm')
    # convert_dataset('book_order')
    # convert_dataset('syntheticdata_nclusters_5_val')
    convert_dataset('syntheticdata_nclusters_5')
    convert_dataset('syntheticdata_nclusters_10')
    convert_dataset('syntheticdata_nclusters_50')
    convert_dataset('syntheticdata_nclusters_100')
