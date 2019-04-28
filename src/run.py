import numpy as np
import torch
import math
from sklearn import metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

anneal_model = {'hrmtpp', 'model11', 'model2'}

def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)

def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def shuffle_data(x_data, t_data):
    shuffled_ind = np.arange(len(x_data))
    np.random.shuffle(shuffled_ind)
    x_data = [x_data[i] for i in shuffled_ind]
    t_data = [t_data[i] for i in shuffled_ind]
    return x_data, t_data


def train(net, params, optimizer, x_data, t_data, label):
    """
    net: PyTorch model class
    params: Namespace class with hyperparameter values
    optimizer: an optimizer object e.g torch.optim.Adam()
    x_data: N, (t_i,D), A list of numpy array.
    t_data: N, (t_i, 2), A list of numpy array.
    """
    net.train()
    N_batches = int(math.ceil(len(x_data) / params.batch_size))
    batch_size = params.batch_size
    # Shuffle the data
    x_data, t_data = shuffle_data(x_data, t_data)
    time_mse, time_mse_count = 0., 0.
    marker_mse, marker_mse_count = 0., 0.
    marker_acc, marker_acc_count = 0., 0.
    total_loss, marker_ll, time_ll = 0., 0, 0.
    total_sequence = 0.

    if params.show:
        from utils.helper import ProgressBar
        bar = ProgressBar(label, max=N_batches)
        
    for b_idx in range(N_batches):
        if params.show: bar.next()
        optimizer.zero_grad()

        unpad_input_x = x_data[b_idx*batch_size:(b_idx+1)*batch_size]
        unpad_input_t = t_data[b_idx*batch_size:(b_idx+1)*batch_size]

        seq_len = [len(lst) for lst in unpad_input_x]
        total_sequence += (sum(seq_len) - len(seq_len))
        max_seq_len = max(seq_len)
        # Shape = T_max_batch x BS x marker_dim

        
        if params.marker_type == 'categorical':
            input_x = np.zeros( (max_seq_len, len(seq_len)) )
        else:
            input_x = np.zeros( (max_seq_len, len(seq_len), unpad_input_x[0].shape[1]) )
        # Shape = T_max_batch x BS x 1
        input_t = np.zeros( (max_seq_len, len(seq_len), params.time_dim) )
        input_mask = np.zeros( (max_seq_len, len(seq_len)) )
        for idx in range(len(seq_len)):
            if params.marker_type == 'categorical':
                input_x[:seq_len[idx], idx ] = unpad_input_x[idx].reshape(-1)    
            else:
                input_x[:seq_len[idx], idx, : ] = unpad_input_x[idx]
            input_t[:seq_len[idx], idx, : ] = unpad_input_t[idx]
            input_mask[:seq_len[idx], idx ] = 1.

        # Convert numpy ndarrays to torch Tensors
        if params.marker_type == 'categorical':
            input_x = torch.from_numpy(input_x).long().to(device)   
        else:
            input_x = torch.from_numpy(input_x).float().to(device)
        input_t = torch.from_numpy(input_t).float().to(device)
        if params.time_scale:
            input_t *= params.time_scale
        input_mask = torch.from_numpy(input_mask).float().to(device)

        #If annealing, pass it here using params.iter
        if params.model in anneal_model:
            anneal = min(1., params.iter/(params.anneal_iter+0.))
            loss, meta_info = net(input_x, input_t, mask=input_mask, anneal = anneal)
        else:
            loss, meta_info = net(input_x, input_t, mask=input_mask)
        loss.backward()

        total_loss += meta_info['true_ll'].numpy()
        marker_ll -= meta_info['marker_ll'].numpy()
        time_ll -= meta_info['time_ll'].numpy()
        
        time_mse += meta_info["time_mse"]
        time_mse_count += meta_info["time_mse_count"]
        if params.marker_type == 'real':
            marker_mse +=  meta_info["marker_mse"]
            marker_mse_count = meta_info["marker_mse_count"]
        else:
            marker_acc+=  meta_info["marker_acc"] 
            marker_acc_count += meta_info["marker_acc_count"]

        
        if params.maxgradnorm >0:
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = params.maxgradnorm)
        optimizer.step()

    optimizer.zero_grad()

    if params.show: bar.finish()

    time_rmse = (time_mse/time_mse_count)** 0.5
    total_loss /= total_sequence
    marker_ll /= total_sequence
    time_ll /= total_sequence
    
    if params.marker_type == 'real':
        marker_rmse = (marker_mse/marker_mse_count)** 0.5
        accuracy = None
        auc = None
    else:
        accuracy = marker_acc/marker_acc_count
        auc = None
        marker_rmse = None
    
    info = {'loss': total_loss, 'time_rmse':time_rmse, 'accuracy': accuracy, 'auc':auc,\
        'marker_rmse': marker_rmse, 'marker_ll':marker_ll, 'time_ll': time_ll}

    return info


def test(net, params,  optimizer,  x_data, t_data, label, dump_cluster = 0):
    """
    x_data: N, (t_i,D), A list of numpy array.
    t_data: N, (t_i, 2), A list of numpy array.
    """
    net.eval()
    N_batches = int(math.ceil(len(x_data) / params.batch_size))
    batch_size = params.batch_size
    # Shuffle the data
    x_data, t_data = shuffle_data(x_data, t_data)
    time_mse, time_mse_count = 0., 0.
    marker_mse, marker_mse_count = 0., 0.
    marker_acc, marker_acc_count = 0., 0.
    total_loss, marker_ll, time_ll = 0., 0, 0.
    total_sequence =0
    if dump_cluster == 1:
        zs = []

    if params.show:
        from utils.helper import ProgressBar
        bar = ProgressBar(label, max=N_batches)
        
    for b_idx in range(N_batches):
        if params.show: bar.next()
        

        unpad_input_x = x_data[b_idx*batch_size:(b_idx+1)*batch_size]
        unpad_input_t = t_data[b_idx*batch_size:(b_idx+1)*batch_size]

        seq_len = [len(lst) for lst in unpad_input_x]
        total_sequence += (sum(seq_len)- len(seq_len))
        max_seq_len = max(seq_len)

        if params.marker_type == 'categorical':
            input_x = np.zeros( (max_seq_len, len(seq_len)) )
        else:
            input_x = np.zeros( (max_seq_len, len(seq_len), unpad_input_x[0].shape[1]) )
        input_t = np.zeros( (max_seq_len, len(seq_len), params.time_dim) )
        input_mask = np.zeros( (max_seq_len, len(seq_len)) )
        for idx in range(len(seq_len)):
            if params.marker_type == 'categorical':
                input_x[:seq_len[idx], idx ] = unpad_input_x[idx].reshape(-1)        
            else:
                input_x[:seq_len[idx], idx, : ] = unpad_input_x[idx]
            input_t[:seq_len[idx], idx,  : ] = unpad_input_t[idx]
            input_mask[:seq_len[idx], idx ] = 1.


        if params.marker_type == 'categorical':
            input_x = torch.from_numpy(input_x).long().to(device)    
        else:
            input_x = torch.from_numpy(input_x).float().to(device)
        input_t = torch.from_numpy(input_t).float().to(device)
        if params.time_scale:
            input_t *= params.time_scale
        input_mask = torch.from_numpy(input_mask).float().to(device)

        with torch.no_grad():
            loss, meta_info = net(input_x, input_t, mask= input_mask)

        total_loss += meta_info['true_ll'].numpy()
        marker_ll -= meta_info['marker_ll'].numpy()
        time_ll -= meta_info['time_ll'].numpy()
        if dump_cluster ==1:
            zs.append(meta_info['z_cluster'].numpy())

        time_mse+= meta_info["time_mse"]
        time_mse_count += meta_info["time_mse_count"]
        if params.marker_type == 'real':
            marker_mse +=  meta_info["marker_mse"]
            marker_mse_count = meta_info["marker_mse_count"]
        else:
            marker_acc+=  meta_info["marker_acc"] 
            marker_acc_count += meta_info["marker_acc_count"]


    if params.show: bar.finish()

    time_rmse = (time_mse/time_mse_count)** 0.5
    if params.marker_type == 'real':
        marker_rmse = (marker_mse/marker_mse_count)** 0.5
        accuracy = None
        auc = None
    else:
        accuracy = marker_acc/marker_acc_count
        auc = None
        marker_rmse = None
    total_loss /= total_sequence
    marker_ll /= total_sequence
    time_ll /= total_sequence
    info = {'loss': total_loss, 'time_rmse':time_rmse, 'accuracy': accuracy, 'auc':auc,\
        'marker_rmse': marker_rmse, 'marker_ll':marker_ll, 'time_ll': time_ll}
    
    if dump_cluster == 1:
        zs = np.concatenate(zs, axis= 1)[0,:,:]
        info['z_cluster'] = zs
    return info

    