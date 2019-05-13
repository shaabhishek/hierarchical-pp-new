import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_time_metric(mu_time,  t, mask, metric_dict):
    if len(mu_time.data.size()) == 3:
        time_mse = ((mu_time[:,:,0]- t[:,:,0])[1:, :] * mask[1:, :]) **2.
        metric_dict['time_mse'] = time_mse.sum().detach().cpu().numpy()
        metric_dict['time_mse_count'] = mask[1:,:].sum().detach().cpu().numpy()
    else: 
        mu_time = torch.mean(mu_time, dim =1)
        time_mse = ((mu_time[:,:,0]- t[:,:,0])[1:, :] * mask[1:, :]) **2.
        metric_dict['time_mse'] = time_mse.sum().detach().cpu().numpy()
        metric_dict['time_mse_count'] = mask[1:,:].sum().detach().cpu().numpy()
    

def get_marker_metric(marker_type, marker_out_mu, x, mask, metric_dict):
    #mask TxBS
    if marker_type == 'real':
        #Compute MSE
        out = (marker_out_mu-x)**2.
        out = out *mask[:,:,None]
        metric_dict['marker_mse'] = out.sum().detach().cpu().numpy()
        metric_dict['marker_mse_count'] = mask.sum().detach().cpu().numpy()
    elif marker_type == 'binary':
        out = (marker_out_mu) >0.5
        true_out = x >0.5
        # acc = (out == true_out)* (mask[:,:,None]== 1.) *true_out
        acc = (out == true_out)*(mask[:,:,None]== 1.)
        metric_dict['marker_acc'] = acc.sum().detach().cpu().numpy()
        # metric_dict['marker_acc_count'] = (true_out * (mask[:,:,None] ==1.)).sum().detach().cpu().numpy()
        metric_dict['marker_acc_count'] = (torch.ones_like(x).to(device)*mask[:,:,None]).sum().cpu().numpy()
    else:
        if len(marker_out_mu.data.size()) == 3:
            out = torch.argmax(marker_out_mu, dim =-1)
            true_out = x 
            # acc = (out == true_out)* (mask[:,:,None]== 1.) *true_out
            acc = (out[1:,:] == true_out[1:,:])*(mask[1:,:]== 1.)
            metric_dict['marker_acc'] = acc.sum().detach().cpu().numpy()
            # metric_dict['marker_acc_count'] = (true_out * (mask[:,:,None] ==1.)).sum().detach().cpu().numpy()
            metric_dict['marker_acc_count'] = (mask[1:,:]).sum().cpu().numpy()
        else: # T x n_sample x BS x dim
            m = torch.nn.Softmax(dim =-1)
            out = m(marker_out_mu) # Txn_sample X BS x dim
            out = torch.mean(out, dim = 1) #T xBS x dim
            out = torch.argmax(out, dim =-1)
            true_out = x 
            acc = (out[1:,:] == true_out[1:,:])*(mask[1:,:]== 1.)
            metric_dict['marker_acc'] = acc.sum().detach().cpu().numpy()
            # metric_dict['marker_acc_count'] = (true_out * (mask[:,:,None] ==1.)).sum().detach().cpu().numpy()
            metric_dict['marker_acc_count'] = (mask[1:,:]).sum().cpu().numpy()


        
        

def compute_point_log_likelihood(model, h, d_js):
        """
            Input:
                h : Tensor of shape TxBSx self.shared_output_layers[-1]
                d_js: Tensor of shape TxBSx1x(N+1) [j,:,:,k] represents the k*delta_x in trapezoidal rule
                [j,:,:,0] is 0
                https://en.wikipedia.org/wiki/Trapezoidal_rule#Uniform_grid
            Output:
                log_f_t : tensor of shape TxBSxC

        """
        #hz_embedded shape TxBSx64, d_js shape is TxBSx1xN
        
        
        #d_js = tj[:, :, 0:1, None]  # Shape TxBSx1xN Time differences

        past_influence = model.h_influence(h)  # TxBSx1,x1


        # TxBSx1x1
        if model.time_influence>0:
            ti = torch.clamp(model.time_influence, min = 1e-5)
        else:
            ti = torch.clamp(model.time_influence, max = -1e-5)
        current_influence = ti[:,:,:, None] * d_js#TxBSx1xN
        base_intensity = model.base_intensity[:,:,:, None]  # 1x1x1x1
        #print(past_influence.shape,current_influence.shape, base_intensity.shape)

        term1 = past_influence + current_influence + base_intensity
        term2 = (past_influence + base_intensity).exp()
        term3 = term1.exp()

        log_f_t = term1 + \
            (1./(ti )) * (term2-term3)
        return log_f_t  # TxBSxCx(N+1)


def compute_time_expectation(model, hz_embedded , t, mask , N = 10000, tol = 0.01, max_try = 3):
    """
        Compute numerical integration.
        Input: 
            model: Torch.nn module
            x : Torch of shape TxBSxmarker_dim
            t : Torch of shape TxBSx2. Last column is indexed by [actual time point, interval]
            mask is needed here.

        Output:
            y : Torch of shape TxBS
    """
    seq_len, batch_size = t.size(0), t.size(1)

    actual_interval = t[:,:,0][:,:,None, None]#TxBSx1x1
    d_max = actual_interval.max()
    init_max = 3.
    delta_x = (d_max* init_max )/N
    d_js = (torch.arange(0, N).float().to(device)*delta_x)[None,None,None,:]#1x1x1xN
    #Add a dummy row for the actual time
    #Interval starts from 0 to max
    repeat_val = (seq_len,batch_size,-1,-1)
    d_js = d_js.expand(*repeat_val)#TxBSx1xN
    if len(hz_embedded.data.size()) == 3:
        hz_embedded = hz_embedded[:,None, :, :]# T, 1, BS, dim
    hz_embedded = hz_embedded.transpose(1,2) #T , BS, 1 , dim

    time_log_likelihood = compute_point_log_likelihood(model, hz_embedded, d_js) #TxBSx1xN  or Tx BS xsample xN

    if time_log_likelihood.size(2) == 1:
        time_likelihood = time_log_likelihood.exp()[:,:,0,:]#TxBSxN (Maybe should be using some stable version of that)
        #Check whether prob sums to 1 or not. If sum< 1, increase integration max limit. If sum>1. Increase N.
        sum_probs = time_likelihood.sum(dim =[2])*delta_x #TxBS
        mean_t_prob = sum_probs.mean()
        #Compute Expectation
        #t*f(t)
        g = time_likelihood * d_js[:,:,0,:]*delta_x# TxBSxN
        expectation = g.sum(dim =-1)#TxBS
        #print("tolerance achieved: ", mean_t_prob-1.)
        return expectation #TxBS
    else: #Number of sample is more
        time_likelihood = time_log_likelihood.exp()#TxBSxSample xN 
        sum_probs = time_likelihood.sum(dim =[3])*delta_x #TxBSxsample
        mean_t_prob = sum_probs.mean()
        #Compute Expectation
        #t*f(t)
        g = time_likelihood * d_js*delta_x# TxBSxsamplexN
        expectation = g.sum(dim =-1)#TxBSxsample
        expectation = expectation.mean(dim = -1)
        #print("tolerance achieved: ", mean_t_prob-1.)
        return expectation #TxBS

