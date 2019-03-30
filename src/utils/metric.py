import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        pass
        #implement categorical
        

def compute_point_log_likelihood(model, h, d_js):
        """
            Input:
                h : Tensor of shape TxBSxCx self.shared_output_layers[-1]
                d_js: Tensor of shape TxBSx1x(N+1) [j,:,:,k] represents the k*delta_x in trapezoidal rule
                [j,:,:,0] is 0
                https://en.wikipedia.org/wiki/Trapezoidal_rule#Uniform_grid
            Output:
                log_f_t : tensor of shape TxBSxC

        """
        #hz_embedded shape TxBSx64, d_js shape is TxBSx1xN
        
        
        #d_js = tj[:, :, 0:1, None]  # Shape TxBSx1xN Time differences

        past_influence = model.h_influence(h)[:,:,:,None]  # TxBSx1,x1


        # TxBSx1x1
        current_influence = model.time_influence * d_js#TxBSx1xN
        base_intensity = model.base_intensity[:,:,:, None]  # 1x1x1x1
        #print(past_influence.shape,current_influence.shape, base_intensity.shape)

        term1 = past_influence + current_influence + base_intensity
        term2 = (past_influence + base_intensity).exp()
        term3 = term1.exp()

        log_f_t = term1 + \
            (1./(model.time_influence+1e-6 )) * (term2-term3)
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

    actual_interval = t[:,:,1][:,:,None, None]#TxBSx1x1
    d_max = actual_interval.max()
    init_max = 3.
    init_N = N
    try_count = 1
    init_tol = tol +1.
    delta_x = (d_max* init_max )/N
    d_js = (torch.arange(0, N).float().to(device)*delta_x)[None,None,None,:]#1x1x1xN
    #Add a dummy row for the actual time
    #Interval starts from 0 to max
    repeat_val = (seq_len,batch_size,-1,-1)
    d_js = d_js.expand(*repeat_val)#TxBSx1xN
    time_log_likelihood = compute_point_log_likelihood(model, hz_embedded, d_js) #TxBSx1xN

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