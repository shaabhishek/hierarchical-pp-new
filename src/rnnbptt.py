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
import matplotlib.pyplot as plt
from base_model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False

#Move it to utils
from utils.metric import get_marker_metric, compute_time_expectation

class rnnbptt(nn.Module):
    """
        Implementation of Recurrent Marked Temporal Point Processes: Embedding Event History to Vector
        'https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf'

        ToDo:
            1. Mask is implemented. Verify
            2. for categorical, x is TxBSx1. create embedding layer with one hot vector

    """

    def __init__(self, marker_type='real', marker_dim=31, time_dim=2, hidden_dim=128, x_given_t=False,base_intensity = 0.,time_influence = 1., gamma = 1., time_loss = 'intensity', dropout=0.):
        super().__init__()
        """
            Input:
                marker_type : 'real' or 'binary', 'categorical'
                marker_dim  : number of dimension  in case of real input. Otherwise number of classes for categorical
                but marker_dim is 1 (just the class label)
                hidden_dim : hidden dimension is gru cell
                x_given_t : whether to condition marker given time gap. For RMTPP set it false.
        """
        self.model_name = 'rnnbptt'#Use this for all model to decide run time behavior
        self.marker_type = marker_type
        self.marker_dim = marker_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.x_given_t = x_given_t
        self.gamma = gamma
        self.time_loss = time_loss
        self.dropout = dropout
        self.use_rnn_cell = True
        self.bptt = 6
        assert_input(self)

        self.sigma_min = 1e-2

        # Set up layer dimensions
        self.x_embedding_layer = [256]
        self.t_embedding_layer = [self.time_dim]
        self.shared_output_layers = [128]
        self.hidden_embed_input_dim = self.hidden_dim 

        # setup layers
        self.embed_x, self.embed_time = create_input_embedding_layer(self)
        if self.use_rnn_cell:
            self.rnn_cell = nn.GRUCell(
                input_size=self.x_embedding_layer[-1], hidden_size=self.hidden_dim)
        else:
            self.rnn = nn.GRU(
                input_size=self.x_embedding_layer[-1],
                hidden_size = self.hidden_dim
                #nonlinearity='relu'
            )
        # For tractivility of conditional intensity time module is a dic where forward needs to be defined
        self.embed_hidden_state, self.output_x_mu, self.output_x_logvar = create_output_marker_layer(self)
        create_output_time_layer(self, base_intensity, time_influence)


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
        
        marker_loss = (-1.* marker_log_likelihood *mask)[1:,:].sum()
        time_loss = (-1. *time_log_likelihood *mask)[1:,:].sum()


        loss = self.gamma*time_loss + marker_loss
        true_loss = time_loss + marker_loss
        #if mask is not None:
        #    loss = loss * mask
        meta_info = {"marker_ll":marker_loss.detach().cpu(), "time_ll":time_loss.detach().cpu(), "true_ll": true_loss.detach().cpu()}
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
        # _, _, phi = preprocess_input(self, x, t)
        phi = self.embed_x(x)

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
                if seq % self.bptt ==0:
                    h_t = self.rnn_cell(phi[seq, :, :], h_t.detach())
                else:
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
        T, bs = x.size(0), x.size(1)

        # marker generation layer. Ideally it should include time gap also.
        # Tensor of shape TxBSx marker_dim
        marker_out_mu, marker_out_logvar = generate_marker(self, 
            hidden_states, t)

        metric_dict = {}
        time_log_likelihood, mu_time = compute_point_log_likelihood(self,
            hidden_states, t)
        with torch.no_grad():
            get_marker_metric(self.marker_type, marker_out_mu, x, mask, metric_dict)
            if self.time_loss == 'intensity':
                expected_t = compute_time_expectation(self, hidden_states, t, mask)
                time_mse = torch.abs(expected_t- t[:,:,0])[1:, :] * mask[1:, :]
            else:
                time_mse = torch.abs(mu_time[:,:,0]- t[:,:,0])[1:, :] * mask[1:, :]
            metric_dict['time_mse'] = time_mse.sum().detach().cpu().numpy()
            metric_dict['time_mse_count'] = mask[1:,:].sum().detach().cpu().numpy()

        
        #Pad initial Time point with 0
        zero_pad = torch.zeros(1, bs).to(device)
        time_log_likelihood = torch.cat([zero_pad, time_log_likelihood[1:,:]], dim =0)
        marker_log_likelihood = compute_marker_log_likelihood(self, 
            x, marker_out_mu, marker_out_logvar)

        return time_log_likelihood, marker_log_likelihood, metric_dict  # TxBS and TxBS

        


if __name__ == "__main__":
    model = rnnbptt()



