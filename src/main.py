import sys
import argparse
import numpy as np
import torch
import os, os.path
import glob
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from run import train, test
from utils.data_loader import load_data, get_dataloader
from utils.model_loader import load_model

def makedir(name):
    try:
        os.makedirs(name)
    except OSError:
        pass

def train_one_dataset(params, file_name, train_dataloader, valid_dataloader):
    """
        Input:
            params: Namespace class with hyperparameter values
            file_name: file to save the model
            train_dataloader: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
            valid_dataloader: list of length num_data_val, each element is numpy array of shape T_i x marker_dim
    """
    ### ================================== model initialization ==================================
    model = load_model(params).to(device)

    if params.l2 >0.:
        optimizer = torch.optim.Adam(model.parameters(), lr = params.lr, weight_decay= params.l2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)

    print(model)
    for parameters in model.parameters():
        print(parameters.size())
    print("\n")

    ### ================================== start training ==================================
    fields = ['loss', 'marker_ll', 'time_ll', 'accuracy',  'time_rmse']
    fs_map = {'loss':'small', 'marker_ll':'large', 'time_ll':'large', 'auc':'large','accuracy':'large', 'marker_rmse':'small', 'time_rmse':'small'}


    datas = ['train', 'valid']
    all_data = {}
    for ds in datas:
        all_data[ds] = {}
        for fs in fields:
            all_data[ds][fs] = {}
            
    best_valid_loss = {}
    best_epoch = {}
    for fs in fields:
        best_epoch[fs] =  1
        best_valid_loss[fs] = None

    for idx in range(params.max_iter):
        params.iter = idx +1
        #### Loss is the ELBO, accuracy is for categorical/binary marker, AUC is for binary/categorical marker.  Time RMSE is w.r.t expectation. marker rmse for real marker####
        train_info = train(model, params, optimizer, train_dataloader,  label='Train')
        valid_info = test(model,  params, optimizer, valid_dataloader, label='Valid')
        print('epoch', idx + 1)
        if params.marker_type != 'real':
            print("valid_auc\t", valid_info['auc'], "\ttrain_auc\t", train_info['auc'])
            print("valid_accuracy\t", valid_info['accuracy'], "\ttrain_accuracy\t", train_info['accuracy'])
        else:
            print("valid_marker_rmse\t", valid_info['marker_rmse'], "\ttrain_marker_rmse\t", train_info['marker_rmse'])
        print("valid_time_mse\t", valid_info['time_rmse'], "\ttrain_time_msee\t", train_info['time_rmse'])
        print("valid_loss\t", valid_info['loss'], "\ttrain_loss\t", train_info['loss'])
        print("valid marker likelihood\t", valid_info['marker_ll'], "\t train marker likelihood\t", train_info['marker_ll'])
        print("valid time likelihood\t", valid_info['time_ll'], "\t train time likelihood\t", train_info['time_ll'])
        
        if not os.path.isdir('model'):
            makedir('model')
        if not os.path.isdir(os.path.join('model', params.save)):
            makedir(os.path.join('model', params.save))
        if not os.path.isdir(os.path.join('model', params.save, params.model)):
            makedir(os.path.join('model', params.save, params.model))
        #net.save_checkpoint(prefix=os.path.join('model', params.save, file_name), epoch=idx+1)
        torch.save({'epoch':idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_info['loss'],
                    },
                     os.path.join('model', params.save, params.model, file_name)+'_'+ str(idx+1)
                    )
        
        for fs in fields:
            all_data['train'][fs][idx+1] =  train_info.get(fs, 0.)
            all_data['valid'][fs][idx+1] =  valid_info.get(fs, 0.)

        # output the epoch with the best validation auc
        for fs in fields:
            if (best_valid_loss[fs] is None) or (fs_map[fs] == 'small' and valid_info[fs] < best_valid_loss[fs]) or \
                (fs_map[fs] == 'large' and valid_info[fs] > best_valid_loss[fs]):
                best_valid_loss[fs] = valid_info[fs]
                best_epoch[fs] = idx+1

    if not os.path.isdir('result'):
        makedir('result')
    if not os.path.isdir(os.path.join('result', params.save)):
        makedir(os.path.join('result', params.save))
    if not os.path.isdir(os.path.join('result', params.save, params.model)):
            makedir(os.path.join('result', params.save, params.model))

    f_save_log = open(os.path.join('result', params.save,params.model,  file_name), 'w')
    for fs in fields:
        for ds in datas:
            f_save_log.write(ds + '_'+ fs +":\n" + str(all_data[ds][fs]) + "\n\n")
    f_save_log.close()
    return best_epoch

def test_one_dataset(params, file_name, test_dataloader, best_epoch, save=False):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    fields = ['loss', 'marker_ll', 'time_ll', 'accuracy',  'time_rmse']
    f_save_log = open(os.path.join('result', params.save,params.model,  file_name), 'a')

    for fs in fields:
        model = load_model(params).to(device)
        checkpoint = torch.load(os.path.join('model', params.save, params.model, file_name)+ '_'+str(best_epoch[fs]))
        model.load_state_dict(checkpoint['model_state_dict'])

        if save and (fs == "loss"):
            if not os.path.isdir('preds'):
                makedir('preds')
            if not os.path.isdir(os.path.join('preds', params.save)):
                makedir(os.path.join('preds', params.save))
            if not os.path.isdir(os.path.join('preds', params.save, params.model)):
                    makedir(os.path.join('preds', params.save, params.model))
            f_save_preds = open(os.path.join('preds', params.save,params.model,  file_name), 'a')
        else:
            f_save_preds = None

        test_info = test(model, params, None, test_dataloader, label='Test', dump_cluster=params.dump_cluster, preds_file=f_save_preds)
        print("test\t ", fs, ': ', test_info[fs])
        print("best epoch of metric: ",fs ,"\t", best_epoch[fs], "All Metrics for that epoch", test_info)
        f_save_log.write('Test results\t :'+ fs  +':\t'+ str(test_info[fs]) + "\n")
        f_save_log.write('Other results for taking best\t :'+ fs  +':\t'+ str(test_info) + "\n")
        if fs == 'loss' and params.dump_cluster ==1:
            f_save_cluster = os.path.join('result', params.save,params.model,  file_name+ '_cluster.pkl')
            with open(f_save_cluster, 'wb') as handle:
                pickle.dump(test_info, handle)


        # if params.marker_type != 'real':
        #     print("\ntest_auc\t", test_info['auc'])
        #     print("test_accuracy\t", test_info['accuracy'])
        # else:
        #     print("test_marker_rmse\t" , test_info['marker_rmse'])
        # print("test_loss\t", test_info['loss'])
        # print("test_time_rmse\t" , test_info['time_rmse'])
        # print("test time likelihood\t", test_info['time_ll'], "\t test marker likelihood\t", test_info['marker_ll'])

    f_save_log.close()
    path = os.path.join('model', params.save, params.model, file_name)+ '*'
    for i in glob.glob(path):
        os.remove(i)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test Marked Point Process.')

    ###Validation Parameter###
    parser.add_argument('--max_iter', type=int, default=1, help='number of iterations')
    parser.add_argument('--anneal_iter', type=int, default=40, help='number of iteration over which anneal goes to 1')
    parser.add_argument('--hidden_dim', type=int, default=256, help='rnn hidden dim')
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
    params = parser.parse_args()
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
    # Read data
    
    #Set Seed for reproducibility
    seedNum = params.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name_identifier = [['_g', params.gamma], ['_do', params.dropout],['_b', params.batch_size],['_h',params.hidden_dim ] , ['_l2', params.l2], ['_l', params.latent_dim], ['_gn', params.maxgradnorm], ['_lr', params.lr], ['_c',params.n_cluster], ['_s',params.seed], ['_tl',params.time_loss], ['_ai', params.anneal_iter]]

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
        best_epoch = train_one_dataset(params, file_name, train_dataloader, valid_dataloader)
        if params.train_test:
            test_data_path = params.data_dir + "/" + params.data_name + '_'+str(params.cv_idx)+ "_test.pkl"
            test_dataloader = get_dataloader(test_data_path, params.marker_type, params.batch_size)
            test_one_dataset(params, file_name, test_dataloader, best_epoch, save=True)
    else:
        test_data_path = params.data_dir + "/" + params.data_name  +'_'+str(params.cv_idx)+"_test.pkl"
        test_dataloader = get_dataloader(test_data_path, params.marker_type, params.batch_size)
        best_epoch = params.best_epoch
        file_name = ''
        for item_ in file_name_identifier:
            file_name = file_name+item_[0]+ str(item_[1])

        test_one_dataset(params, file_name, test_dataloader, best_epoch)
