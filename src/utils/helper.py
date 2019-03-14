from progress.bar import Bar
import numpy as np

class ProgressBar(Bar):
    message = 'Loading'
    fill = '='
    suffix = '%(percent).1f%% | Elapsed: %(elapsed)ds | ETA: %(eta)ds '


def train_val_split(data, val_ratio=0.2):
    """
        Input:
            data: dict with keys 'x' and 't'.
            data['x']: x_data: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
            data['t']: t_data: list of length num_data_train, each element is numpy array of shape T_i x 3
        Output:
            train_split, val_split: both follow the same structure as data
    """
    N = len(data['x'])
    val_size = int(val_ratio * N)
    
    random_order = np.arange(N)
    np.random.shuffle(random_order)
    
    train_split = {}
    val_split = {}
    for key, value in data.items():
        train_split[key] = [data[key][i] for i in random_order[:N-val_size]]
        val_split[key] = [data[key][i] for i in random_order[N-val_size:]]

    return train_split, val_split