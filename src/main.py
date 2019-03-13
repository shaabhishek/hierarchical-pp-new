import sys
import os
import argparse
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from run import train, test
from utils.data_loader import load_data
from utils.model_loader import load_model



def train_one_dataset(params, file_name, train_x_data, train_t_data, valid_x_data, valid_t_data):
    ### ================================== model initialization ==================================
    model = load_model(params).to(device)

    if params.reg == 'l2':
        optimizer = torch.optim.Adam(model.parameters(), lr = params.lr, weight_decay= params.l2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)

    for parameters in model.parameters():
        print(parameters.size())
    print("\n")

    ### ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    all_valid_marker_rmse = {}
    all_train_marker_rmse = {}
    all_valid_time_rmse = {}
    all_train_time_rmse = {}
    best_valid_loss = 0

    for idx in range(params.max_iter):
        params.iter = idx +1
        #### Loss is the ELBO, accuracy is for categorical/binary marker, AUC is for binary/categorical marker.  Time RMSE is w.r.t expectation. marker rmse for real marker####
        train_loss, train_time_rmse, train_accuracy, train_auc, train_marker_rmse = train(model, params, optimizer, train_x_data, train_t_data,  label='Train')
        valid_loss, valid_time_rmse, valid_accuracy, valid_auc, valid_marker_rmse = test(model,  params, optimizer, valid_x_data, valid_t_data, label='Valid')

        print('epoch', idx + 1)
        if params.marker_type != 'real':
            print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
            print("valid_accuracy\t", valid_accuracy, "\ttrain_accuracy\t", train_accuracy)
        else:
            print("valid_marker_rmse\t", valid_marker_rmse, "\ttrain_marker_rmse\t", train_marker_rmse)
        print("valid_marker_rmse\t", valid_time_rmse, "\ttrain_time_rmse\t", train_time_rmse)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)

        if not os.path.isdir('model'):
            os.makedirs('model')
        if not os.path.isdir(os.path.join('model', params.save)):
            os.makedirs(os.path.join('model', params.save))
        if not os.path.isdir(os.path.join('model', params.save, params.model_name)):
            os.makedirs(os.path.join('model', params.save, params.model_name))
        #net.save_checkpoint(prefix=os.path.join('model', params.save, file_name), epoch=idx+1)
        torch.save({'epoch':idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    },
                     os.path.join('model', params.save, params.model_name, file_name)+'_'+ str(idx+1)
                    )

        if params.marker_type != 'real':
            all_valid_auc[idx + 1] = valid_auc
            all_train_auc[idx + 1] = train_auc
            all_valid_accuracy[idx + 1] = valid_accuracy
            all_train_accuracy[idx + 1] = train_accuracy
        else:
            all_valid_marker_rmse[idx + 1] = valid_marker_rmse
            all_train_marker_rmse[idx + 1] = train_marker_rmse
        all_valid_time_rmse[idx + 1] = valid_time_rmse
        all_train_time_rmse[idx + 1] = train_time_rmse
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        
        

        # output the epoch with the best validation auc
        if valid_loss > best_valid_loss :
            best_valid_loss = valid_loss
            best_epoch = idx+1

    if not os.path.isdir('result'):
        os.makedirs('result')
    if not os.path.isdir(os.path.join('result', params.save)):
        os.makedirs(os.path.join('result', params.save))
    if not os.path.isdir(os.path.join('result', params.save, params.model_name)):
            os.makedirs(os.path.join('result', params.save, params.model_name))

    f_save_log = open(os.path.join('result', params.save,params.model_name,  file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.write("valid_marker_rmse:\n" + str(all_valid_marker_rmse) + "\n\n")
    f_save_log.write("train_marker_rmse:\n" + str(all_train_marker_rmse) + "\n\n")
    f_save_log.write("valid_time_rmse:\n" + str(all_valid_time_rmse) + "\n\n")
    f_save_log.write("train_time_rmse:\n" + str(all_train_time_rmse) + "\n\n")
    f_save_log.close()
    return best_epoch

def test_one_dataset(params, file_name, test_x_data, test_t_data, best_epoch):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params).to(device)


    checkpoint = torch.load(os.path.join('model', params.save, params.model_name, file_name)+ '_'+str(best_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_time_rmse, test_accuracy, test_auc, test_marker_rmse = test(model, params, test_x_data, test_t_data, label='Test')
    print("\ntest_auc\t", test_auc)
    print("test_accuracy\t", test_accuracy)
    print("test_loss\t", test_loss)
    print("train_marker_rmse\t" + test_marker_rmse)
    print("train_time_rmse\t" + test_time_rmse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test Marked Point Process.')

    ###Validation Parameter###
    parser.add_argument('--max_iter', type=int, default=200, help='number of iterations')
    parser.add_argument('--anneal_iter', type=int, default=100, help='number of iteration over which anneal goes to 1')
    parser.add_argument('--hidden_dim', type=int, default=128, help='rnn hidden dim')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-3, help='regularizer with weight decay parameter')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('--latent_dim', type=int, default=20, help='latent dim')
    parser.add_argument('--x_given_t', type=bool, default=False, help='whether x given t')
    parser.add_argument('--reg', type=str, default='l2', help='regularization')
    parser.add_argument('--n_cluster', type=int, default=10, help='number of cluster')
    

    ###Helper Parameter###
    parser.add_argument('--model', type=str, default='rmtpp', help='model name')
    parser.add_argument('--data_name', type=str, default='mimic', help='data set name')

    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--data_dir', type=str, default='../data/', help='data directory')
    parser.add_argument('--best_epoch', type=int, default=10, help='best epoch')





    params = parser.parse_args()
    ###Fixed parameter###
    if params.data_name == 'mimic':
        params.marker_dim = 20
        params.marker_type = 'binary'
        parser.load = 'mimic'
        parser.save = 'mimic'

    else:#different dataset. Encode those details.
        pass

    
    # Read data
    
    #Set Seed for reproducibility
    seedNum =1337
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)

    if not params.test:
        d = vars(params)
        for key in d:
            print('\t', key, '\t', d[key])
        file_name = 'b' + str(params.batch_size) + \
                    '_gn' + str(params.maxgradnorm) + '_h' + str(params.hidden_dim) + \
                    '_l2' + str(params.l2) + '_l' + str(params.latent_dim) + \
                    '_lr' + str(params.init_lr) + '_c' + str(params.n_cluster) + \
                    '_xt' + str(params.x_given_t) + '_r' + str(params.reg) + \
                    '_s'+str(seedNum) 

        #Data should reside in this path for all datasets. Ideally 5 cross fold validation.
        train_data_path = params.data_dir + "/" + params.data_name + "_train1.pkl"
        valid_data_path = params.data_dir + "/" + params.data_name + "_valid1.pkl"
        #That pkl file should give two list of x and t. It should not be tensor.
        train_x_data, train_t_data = load_data(train_data_path)
        valid_x_data, valid_t_data = load_data(valid_data_path)
        print("\n")
        print("train data length", len(train_x_data))
        print("valid data length", len(valid_x_data))
        print("\n")
        best_epoch = train_one_dataset(params, file_name, train_x_data, train_t_data, valid_x_data, valid_t_data)
        if params.train_test:
            test_data_path = params.data_dir + "/" + params.data_name + "_test1.pkl"
            test_x_data, test_t_data = load_data(test_data_path)
            test_one_dataset(params, file_name, test_x_data, test_t_data, best_epoch)
    else:
        test_data_path = params.data_dir + "/" + params.data_name  +"_test1.pkl"
        test_x_data, test_t_data = load_data(test_data_path)
        best_epoch = params.best_epoch
        file_name = 'b' + str(params.batch_size) + \
                    '_gn' + str(params.maxgradnorm) + '_h' + str(params.hidden_dim) + \
                    '_l2' + str(params.l2) + '_l' + str(params.latent_dim) + \
                    '_lr' + str(params.init_lr) + '_c' + str(params.n_cluster) + \
                    '_xt' + str(params.x_given_t) + '_r' + str(params.reg) + \
                    '_s'+str(seedNum)
        test_one_dataset(params, file_name, test_x_data, test_t_data, best_epoch)