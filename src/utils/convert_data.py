import numpy as np
import pickle
from helper import train_val_split

def convert_dataset(data_name):
    if data_name not in {'lastfm'}:
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
                           times.append(values/3600*24.)
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
                time_data_path = './../data/real/'+data_name+'/'+ 'event_split_1000'+'-'+ls+'.txt'
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





if __name__ == "__main__":
    convert_dataset('mimic2')
    convert_dataset('so')
    convert_dataset('lastfm')
    convert_dataset('book_order')