import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def compute_time_expectation(model, x, t, mask = None, N = 1000, tol = 0.02, max_try = 5):
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
    seq_len, batch_size = x.size(0), x.size(1)

    #Instead of x,t pass hz_embedded which can be used for both marker comutation as well as time computation
    hz_embedded = model.compute_hidden_states(x,t, mask)

    actual_interval = t[:,:,1][:,:,None, None]#TxBSx1x1
    d_max = actual_interval.max()
    init_max = 2.
    init_N = N
    try_count = 1
    init_tol = tol +1.
    while(try_count<max_try and init_tol<tol):
        try_count += 1
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
        if(mean_t_prob>1.+tol and try_count<max_try):
            init_N *= 2
            continue
        elif(mean_t_prob<1.-tol and try_count<max_try):
            init_max *=2.
            continue
        #Compute Expectation
        #t*f(t)
        g = time_likelihood * d_js[:,:,0,:]*delta_x# TxBSxN
        expectation = g.sum(dim =-1)#TxBS
        print("tolerance achieved: ", mean_t_prob-1.)
        return expectation #TxBS





        
        

 




