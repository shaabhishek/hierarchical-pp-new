import argparse
from argparse import Namespace

import numpy as np
import torch

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(params:Namespace, split_name):
    split_name = "train" #TODO remove after testing
    #Data should reside in this path for all datasets. Ideally 5 cross fold validation.
    data_path = params.data_dir + params.data_name +'_'+str(params.cv_idx)+ f"_{split_name}.pkl"
    dataloader = get_dataloader(data_path, params.marker_type, params.batch_size)
    return dataloader

def load_model(params:Namespace):
    model = load_model(params).to(device)
    return model

def _augment_params(params:Namespace):
    params.cv_idx = 1

    ###Fixed parameter###
    if params.data_name == 'mimic2':
        params.marker_dim = 75
        params.base_intensity = -0.
        params.time_influence = 1.
        params.time_dim = 2
        params.marker_type = 'categorical'
        params.batch_size = 32

    elif params.data_name == 'so':
        params.marker_dim = 22
        params.time_dim = 2
        params.base_intensity = -5.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32

    elif params.data_name == 'meme':
        params.marker_dim = 5000
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 64
        params.time_scale = 1e-3

    
    elif params.data_name == 'retweet':
        params.marker_dim = 3
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32
        params.time_scale = 1e-3

    elif params.data_name == 'book_order':
        params.marker_dim = 2
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 10
    
    elif params.data_name == 'lastfm':
        params.marker_dim = 3150
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32
    
    elif params.data_name == 'simulated_hawkes':
        params.marker_dim = 3150
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32

    elif 'syntheticdata' in params.data_name:
        params.marker_dim = 2
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32


    else:#different dataset. Encode those details.
        raise ValueError

class Plotter():
    def __init__(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to test Marked Point Process.')
    
    parser.add_argument('--model', type=str, default='model2', help='model name')
    parser.add_argument('--data_name', type=str, default='mimic2', help='data set name')

    params = parser.parse_args()

    _augment_params(params)

    if params.time_loss == 'intensity':
        params.n_sample = 1
    if params.time_loss == 'normal':
        params.n_sample = 5

    params.load = params.data_name

    loader = load_data(params, "train")
    model = load_model(params)