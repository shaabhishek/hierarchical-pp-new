import sys
import argparse
from argparse import Namespace
import numpy as np
import torch
import os, os.path
import glob
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from run import train, test
from utils.data_loader import load_data, get_dataloader
from utils.model_loader import load_model, ModelLoader
from utils.logger import Logger

def makedir(name):
    try:
        os.makedirs(name)
    except OSError:
        pass

def checkpoint_model(model, optimizer, epoch_num, loss, params, file_name):
    state = {
        'epoch': epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    path = os.path.join('model', params.save, params.model, file_name)
    torch.save(state, path)

def train_one_dataset(params, file_name, train_dataloader, valid_dataloader, logger):
    """
        Input:
            params: Namespace class with hyperparameter values
            file_name: file to save the model
            train_dataloader: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
            valid_dataloader: list of length num_data_val, each element is numpy array of shape T_i x marker_dim
    """
    ### ================================== model initialization ==================================
    model = load_model(params).to(device)
    model.print_info()

    if params.l2 >0.:
        optimizer = torch.optim.Adam(model.parameters(), lr = params.lr, weight_decay= params.l2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)


    ### ================================== start training ==================================

    for idx in range(1, params.max_iter+1):
        params.iter = idx
        # Loss is the ELBO
        # Accuracy is for categorical/binary marker,
        # AUC is for binary/categorical marker.
        # Time RMSE is w.r.t expectation.
        # Marker rmse for real marker####
        train_info = train(model, params, optimizer, train_dataloader,  label='Train')
        valid_info = test(model,  params, valid_dataloader, label='Valid')

        ### ================== start epoch logging ====================
        # print train and validation metric values
        logger.print_train_epoch(idx, train_info, valid_info)

        # save train and validation metric values
        logger.log_train_epoch(idx, train_info, valid_info)
        ### ================== finish epoch logging ====================

        # saves the model, optimizer and loss as a checkpoint
        if logger.get_best_epoch(metric_name='loss') == idx:
            checkpoint_model(model, optimizer, idx, train_info['loss'], params, file_name)

    ### ================================== finish training ==================================

    # End of training: save the logger state (metric values) to file
    for split in ['train', 'valid']:
        logger.save_logs_to_file(split)

    print(f"Model saved at {os.path.join('model', params.save, params.model, file_name)}")

def test_one_dataset(params, file_name, test_dataloader, logger:Logger, save=False):
    print("\n\nStart testing ......................\nBest epoch:", logger.best_epoch)
    best_epoch_num = logger.get_best_epoch(metric_name='loss')

    ### ================================== start setting up state ==================================

    # Load checkpointed model
    # model = load_model(params).to(device)
    model_state_path = os.path.join('model', params.save, params.model, file_name)
    model = ModelLoader(params, model_state_path=model_state_path).model

    # checkpoint = torch.load(model_state_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    ### ================================== start testing ==================================

    predictions_save_path = os.path.join('preds', params.save,params.model, file_name)
    with open(predictions_save_path, 'a') as f_save_preds:
        # Saving predictions performed by the model itself
        test_info = test(model, params, test_dataloader, label='Test', dump_cluster=params.dump_cluster, preds_file=f_save_preds)

    ### ================================== finish testing ==================================

    # Print the test metric info
    logger.print_test_epoch(test_info)

    # Log the test metric info
    logger.log_test_epoch(best_epoch_num, test_info)

    # End of epoch: save the logger state (metric values) to file
    logger.save_logs_to_file('test')
    
    ### ================================== finish logging ==================================


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

    if params.time_loss == 'intensity':
        params.n_sample = 1
    if params.time_loss == 'normal':
        params.n_sample = 5

    params.load = params.data_name
    params.save = params.data_name
    return params

def setup_parser():
    parser = argparse.ArgumentParser(description='Script to test Marked Point Process.')

    ###Validation Parameter###
    parser.add_argument('--max_iter', type=int, default=1, help='number of iterations')
    parser.add_argument('--anneal_iter', type=int, default=40, help='number of iteration over which anneal goes to 1')
    parser.add_argument('--rnn_hidden_dim', type=int, default=256, help='rnn hidden dim')
    parser.add_argument('--maxgradnorm', type=float, default=10.0, help='maximum gradient norm')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='tradeoff of time and marker in loss. marker loss + gamma * time loss')
    parser.add_argument('--l2', type=float, default=0., help='regularizer with weight decay parameter')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('--latent_dim', type=int, default=20, help='latent dim')
    parser.add_argument('--x_given_t', type=bool, default=False, help='whether x given t')
    parser.add_argument('--n_cluster', type=int, default=10, help='number of cluster')


    ###Helper Parameter###
    parser.add_argument('--model', type=str, default='model2', help='model name')
    parser.add_argument('--time_loss', type=str, default='intensity', help='whether to use normal loss or intensity loss')
    parser.add_argument('--time_scale', type=float, default=1, help='scaling factor to multiply the timestamps with')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--data_dir', type=str, default='../data/', help='data directory')
    parser.add_argument('--best_epoch', type=int, default=10, help='best epoch')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--dump_cluster', type=int, default=0, help='whether to dump cluster while Testing')

    parser.add_argument('--data_name', type=str, default='mimic2', help='data set name')

    return parser

if __name__ == '__main__':
    parser = setup_parser()
    params = parser.parse_args()
    params = _augment_params(params)

    # Read data

    #Set Seed for reproducibility
    seedNum = params.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name_identifier = [['_g', params.gamma], ['_do', params.dropout],['_b', params.batch_size],['_h',params.rnn_hidden_dim ] , ['_l2', params.l2], ['_l', params.latent_dim], ['_gn', params.maxgradnorm], ['_lr', params.lr], ['_c',params.n_cluster], ['_s',params.seed], ['_tl',params.time_loss], ['_ai', params.anneal_iter]]

    if not params.test:
        d = vars(params)
        for key in d:
            print('\t', key, '\t', d[key])
        file_name = ''
        for item_ in file_name_identifier:
            file_name = file_name+item_[0]+ str(item_[1])

        #Data should reside in this path for all datasets. Ideally 5 cross fold validation.
        train_data_path = params.data_dir + params.data_name +'_'+str(params.cv_idx)+ "_train.pkl"
        valid_data_path = params.data_dir + params.data_name + '_'+str(params.cv_idx)+"_test.pkl"
        #That pkl file should give two list of x and t. It should not be tensor.
        train_dataloader = get_dataloader(train_data_path, params.marker_type, params.batch_size)
        valid_dataloader = get_dataloader(valid_data_path, params.marker_type, params.batch_size)

        # train_x_data, train_t_data = load_data(train_data_path)
        # valid_x_data, valid_t_data = load_data(valid_data_path)
        print("\n")
        print("train data length", len(train_dataloader.dataset))
        print("valid data length", len(valid_dataloader.dataset))
        print("\n")

        logs_save_path = os.path.join('result', params.save, params.model, file_name)
        logger = Logger(marker_type=params.marker_type, dataset_name=params.save, model_name=params.model, logs_save_path=logs_save_path)
        train_one_dataset(params, file_name, train_dataloader, valid_dataloader, logger)
        if params.train_test:
            test_data_path = params.data_dir + "/" + params.data_name + '_'+str(params.cv_idx)+ "_test.pkl"
            test_dataloader = get_dataloader(test_data_path, params.marker_type, params.batch_size)
            test_one_dataset(params, file_name, test_dataloader, logger, save=True)
    else:
        test_data_path = params.data_dir + "/" + params.data_name  +'_'+str(params.cv_idx)+"_test.pkl"
        test_dataloader = get_dataloader(test_data_path, params.marker_type, params.batch_size)
        best_epoch = params.best_epoch
        file_name = ''
        for item_ in file_name_identifier:
            file_name = file_name+item_[0]+ str(item_[1])

        test_one_dataset(params, file_name, test_dataloader, best_epoch)
