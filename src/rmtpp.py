import torch
import numpy as np
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam
#import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False

#Move it to utils
from utils.metric import get_marker_metric, compute_time_expectation





def one_hot_encoding(y, n_dims=None):
    # Implement
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


class rmtpp(nn.Module):
    """
        Implementation of Recurrent Marked Temporal Point Processes: Embedding Event History to Vector
        'https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf'

        ToDo:
            1. Mask is implemented. Verify
            2. for categorical, x is TxBSx1. create embedding layer with one hot vector

    """

    def __init__(self, marker_type='real', marker_dim=31, time_dim=2, hidden_dim=128, x_given_t=False, ):
        super().__init__()
        """
            Input:
                marker_type : 'real' or 'binary', 'categorical'
                marker_dim  : number of dimension  in case of real input. Otherwise number of classes for categorical
                but marker_dim is 1 (just the class label)
                hidden_dim : hidden dimension is gru cell
                x_given_t : whether to condition marker given time gap. For RMTPP set it false.
        """
        self.model_name = 'rmtpp'#Use this for all model to decide run time behavior
        self.marker_type = marker_type
        self.marker_dim = marker_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.x_given_t = x_given_t
        self.use_rnn_cell = False
        self.assert_input()

        self.sigma_min = 1e-2

        # Set up layer dimensions
        self.x_embedding_layer = [64]
        self.t_embedding_layer = [64]
        self.shared_output_layers = [64]

        # setup layers
        self.embed_x, self.embed_time = self.create_input_embedding_layer()
        if self.use_rnn_cell:
            self.rnn_cell = nn.GRUCell(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
        else:
            self.rnn = nn.GRU(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1],
                hidden_size = self.hidden_dim
                #nonlinearity='relu'
            )
        # For tractivility of conditional intensity time module is a dic where forward needs to be defined
        self.embed_hidden_state, self.output_x_mu, self.output_x_logvar = self.create_output_marker_layer()
        self.h_influence, self.time_influence, self.base_intensity = self.create_output_time_layer()

    def assert_input(self):
        assert self.marker_type in {
            'real', 'categorical', 'binary'}, "Unknown Input type provided!"

    def create_input_embedding_layer(self):
        x_module = nn.Sequential(
            nn.Linear(self.marker_dim, self.x_embedding_layer[0])#, nn.ReLU(),
            # Not sure whether to put Relu at the end of embedding layer
            #nn.Linear(self.x_embedding_layer[0],
            #          self.x_embedding_layer[1]), nn.ReLU()
        )

        #t_module = nn.Linear(self.time_dim, self.t_embedding_layer[0])
        t_module = nn.Sequential(
            nn.Linear(self.time_dim, self.t_embedding_layer[0]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.t_embedding_layer[0], self.t_embedding_layer[0])
        )
        return x_module, t_module

    def create_output_marker_layer(self):
        embed_module = nn.Sequential(
            nn.Linear(self.hidden_dim,
                      self.shared_output_layers[0]), nn.ReLU(),
            #nn.Linear(
            #    self.shared_output_layers[0], self.shared_output_layers[1]), nn.ReLU()
        )

        x_module_logvar = None
        l = self.shared_output_layers[-1]
        if self.x_given_t:
            l += 1
        if self.marker_type == 'real':
            x_module_mu = nn.Linear(l, self.marker_dim)
            x_module_logvar = nn.Linear(l, self.marker_dim)
        elif self.marker_type == 'binary':#Fix binary
            x_module_mu = nn.Sequential(
                nn.Linear(l, self.marker_dim),
                nn.Sigmoid())
            

        return embed_module, x_module_mu, x_module_logvar

    def create_output_time_layer(self):

        h_influence =  nn.Linear(self.shared_output_layers[-1], 1, bias=False)
        time_influence = nn.Parameter(0.005*torch.ones(1, 1, 1))
        base_intensity =  nn.Parameter(torch.zeros(1, 1, 1)-8.)
        return h_influence, time_influence, base_intensity

    def forward(self, x, t,anneal = 1., mask= None):
        """
            Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSxtime_dim. [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
                mask: Tensor of shape TxBS If mask[t,i] =1 then that timestamp is present
            Output:
                loss : Tensor scalar
                meta_info : dict of results
        """
        #TxBS and TxBS
        time_log_likelihood, marker_log_likelihood, metric_dict = self._forward(
            x, t, mask)
        if DEBUG:
            print("Losses:", marker_log_likelihood.sum().item(),  time_log_likelihood.sum().item())
        loss = -1. * (time_log_likelihood + marker_log_likelihood)
        if mask is not None:
            loss = loss * mask
        loss = loss.sum()
        meta_info = {"marker_ll":-marker_log_likelihood.sum().detach().cpu(), "time_ll":-time_log_likelihood.sum().detach().cpu(), "true_ll":  -loss.detach().cpu()}
        return loss, {**meta_info, **metric_dict}

    def run_forward_rnn(self, x, t):
        """
            Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
            Output:
                h : Tensor of shape (T+1)xBSxhidden_dim
                embed_h : Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
        """
        batch_size, seq_length = x.size(1), x.size(0)
        # phi Tensor shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
        _, _, phi = self.preprocess_input(x, t)

        if self.use_rnn_cell is False:
            # Run RNN over the concatenated sequence [marker_seq_emb, time_seq_emb]
            h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
            hidden_seq, _ = self.rnn(phi, h_0)
            h = torch.cat([h_0, hidden_seq], dim = 0)

        else:
            outs = []
            h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
            outs.append(h_t[None, :, :])
            for seq in range(seq_length):
                h_t = self.rnn_cell(phi[seq, :, :], h_t)
                outs.append(h_t[None, :, :])
            h = torch.cat(outs, dim=0)  # shape = [T+1, batchsize, h]
        return h[:-1,:,:], self.preprocess_hidden_state(h)[:-1,:,:]

    def preprocess_hidden_state(self, h):
        return self.embed_hidden_state(h)

    def compute_hidden_states(self, x, t, mask):
        """
        Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSxtime_dim. [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
                
                mask: does not matter
        Output:
                hz : Tensor of shape (T)xBSxself.shared_output_layers[-1]
        """
        _, hidden_states = self.run_forward_rnn(x, t)
        return hidden_states


    def preprocess_input(self, x, t):
        """
            Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSxtime_dim. [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}

            Output:
                phi_x : Tensor of shape TxBSx self.x_embedding_layer[-1]
                phi_t : Tensor of shape TxBSx self.t_embedding_layer[-1]
                phi   : Tensor of shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
        """
        if self.marker_type == 'categorical':
            # Shape TxBSxmarker_dim
            x = one_hot_encoding(x[:, :, 0], self.marker_dim)
        phi_x = self.embed_x(x)
        phi_t = self.embed_time(t)
        phi = torch.cat([phi_x, phi_t], -1)
        return phi_x, phi_t, phi

    def _forward(self, x, t, mask):
        """
            Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
                mask: Tensor of shape TxBSx1. If mask[t,i,0] =1 then that timestamp is present
            Output:

        """

        # Tensor of shape (T)xBSxself.shared_output_layers[-1]
        _, hidden_states = self.run_forward_rnn(x, t)

        # marker generation layer. Ideally it should include time gap also.
        # Tensor of shape TxBSx marker_dim
        marker_out_mu, marker_out_logvar = self.generate_marker(
            hidden_states, t)

        metric_dict = {}
        with torch.no_grad():
            get_marker_metric(self.marker_type, marker_out_mu, x, mask, metric_dict)
            expected_t = compute_time_expectation(self, hidden_states, t, mask)
            time_mse = (expected_t- t[:,:,1])**2. * mask
            metric_dict['time_mse'] = time_mse.sum().detach().cpu().numpy()
            metric_dict['time_mse_count'] = mask.sum().detach().cpu().numpy()

        time_log_likelihood = self.compute_point_log_likelihood(
            hidden_states, t)
        marker_log_likelihood = self.compute_marker_log_likelihood(
            x, marker_out_mu, marker_out_logvar)

        return time_log_likelihood, marker_log_likelihood, metric_dict  # TxBS and TxBS
    

    def compute_marker_log_likelihood(self, x, mu, logvar):
        """
            Input:  
                    x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                    mu : Tensor of shape T x BS x marker_dim
                    logvar : Tensor of shape T x BS x marker_dim #None in case of non real marker
            Output:
                    loss : TxBS
        """
        if self.marker_type == 'real':
            sigma = torch.clamp(logvar.exp().sqrt(), min= self.sigma_min)
            x_recon_dist = Normal(mu, sigma)
            ll_loss = (x_recon_dist.log_prob(x)
                        ).sum(dim=-1)
            return ll_loss
        else:
            seq_lengths, batch_size = x.size(0), x.size(1)
            
            if  self.marker_type == 'categorical':
                mu_ = mu.view(-1, self.marker_dim)  # T*BS x marker_dim
                x_ = x.view(-1)  # (T*BS,)
                loss = F.cross_entropy(mu_, x_, reduction='none').view(
                    seq_lengths, batch_size)
            else:#binary
                loss = F.binary_cross_entropy(mu, x, reduction= 'none').sum(dim =-1)#TxBS
            return -loss

    def compute_point_log_likelihood(self, h, t):
        """
            Input:
                h : Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
                t : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
            Output:
                log_f_t : tensor of shape TxBS

        """
        h_trimmed = h # TxBSxself.shared_output_layers[-1]
        d_js = t[:, :, 1][:, :, None]  # Shape TxBSx1 Time differences

        past_influence = self.h_influence(h_trimmed)  # TxBSx1

        # TxBSx1
        current_influence = self.time_influence * d_js
        base_intensity = self.base_intensity  # 1x1x1

        term1 = past_influence + current_influence + base_intensity
        term2 = (past_influence + base_intensity).exp()
        term3 = term1.exp()

        log_f_t = term1 + \
            (1./(self.time_influence+1e-6)) * (term2-term3)
        return log_f_t[:, :, 0]  # TxBS

    def generate_marker(self, h, t):
        """
            Input:
                h : Tensor of shape TxBSxself.shared_output_layers[-1]
                t : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
            Output:
                marker_out_mu : Tensor of shape T x BS x marker_dim
                marker_out_logvar : Tensor of shape T x BS x marker_dim #None in case of non real marker
        """
        # h_trimmed = h[:-1, :, :]
        h_trimmed = h
        if self.x_given_t:
            d_js = t[:, :, 1][:, :, None]  # Shape TxBSx1 Time differences
            h_trimmed = torch.cat([h_trimmed, d_js], -1)
        
        marker_out_mu = self.output_x_mu(h_trimmed)

        if self.marker_type == 'real':
            marker_out_logvar = self.output_x_logvar(h_trimmed)
        else:
            marker_out_logvar = None
        return marker_out_mu, marker_out_logvar

        


if __name__ == "__main__":
    model = rmtpp()
    # data = generate_mpp()
    data = mimic_data_tensors()


