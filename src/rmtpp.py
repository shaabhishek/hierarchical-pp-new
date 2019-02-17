import torch
import numpy as np
import random
# import torch.utils.data
# from torchvision import datasets, transforms
import utils
import time
# from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam
import matplotlib.pyplot as plt
# import pdb; pdb.set_trace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot_encoding(x):
    #Implement
    return x
class rmtpp(nn.Module):
    """
        Implementation of Recurrent Marked Temporal Point Processes: Embedding Event History to Vector
        'https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf'

        ToDo:
            1. Mask is implemented. Verify
            2. for categorical, x is TxBSx1. create embedding layer with one hot vector

    """
    def __init__(self, marker_type ='real', marker_dim=21, hidden_dim=300, x_given_t = False ):
        """
            Input:
                marker_type : 'real' or 'binary', 'categorical'
                marker_dim  : number of dimension  in case of real input. Otherwise number of classes for categorical
                but marker_dim is 1 (just the class label)
                hidden_dim : hidden dimension is gru cell
                x_given_t : whether to condition marker given time gap. For RMTPP set it false.
        """
        self.marker_type = marker_type
        self.marker_dim  = marker_dim
        self.hidden_dim = hidden_dim
        self.x_given_t = x_given_t
        self.assert_input()

        #Set up layer dimensions
        self.x_embedding_layer = [ 64, 64]
        self.t_embedding_layer = [16]
        self.shared_output_layers = [ 64, 64]

        #setup layers
        self.embed_x, self.embed_time = self.create_input_embedding_layer()
        self.rnn_cell = nn.GRUCell(input_size=self.self.x_embedding_layer[-1] + self.t_embedding_layer[-1]\
            , hidden_size=self.hidden_dim)
        #For tractivility of conditional intensity time module is a dic where forward needs to be defined
        self.embed_hidden_state, self.output_x_mu, self.output_x_logvar, self.output_time_dic = self.create_output_layer()


        def assert_input(self):
            assert self.marker_type in {'real', 'categorical', 'binary'}, "Unknown Input type provided!"
            if self.marker_type == 'binary' and marker_dim != 2:
                self.marker_dim = 2
                print("Setting marker dimension to 2 for binary input!")

        def create_input_embedding_layer(self):
            x_module = nn.Sequential(
                nn.Linear(self.marker_dim, self.x_embedding_dim[0]), nn.ReLU(),
                #Not sure whether to put Relu at the end of embedding layer
                nn.Linear(self.x_embedding_dim[0], self.x_embedding_dim[1]), nn.ReLU()
            )

            t_module = nn.Linear(2, self.t_embedding_layer[0])
            return x_module, t_module

        def create_output_layer(self):
            embed_module = nn.Sequential(
                nn.Linear(self.hidden_dim, self.shared_output_layers[0]), nn.ReLU(),
                nn.Linear(self.shared_output_layers[0], self.shared_output_layers[1]), nn.ReLU()
            )

            x_module_logvar = None
            l = self.shared_output_layers[-1]
            if self.x_given_t:
                l += 1
            x_module_mu = nn.Linear(l, self.marker_dim)
            if self.marker_type == 'real':
                x_module_logvar = nn.Linear(l, self.marker_dim)

            

            time_module_dic = nn.ModuleDict({
                'h_influence': nn.Linear(shared_output_layers[-1], 1, bias = False),
                'time_influence': nn.Parameter(torch.zeros(1,1,1)),
                'base_intensity': nn.Parameter(torch.zeros(1, 1,1))
                })
            return embed_module, x_module_mu, x_module_logvar, time_module_dic



        def forward(self, x, t, mask  = None):
            """
                Input: 
                    x   : Tensor of shape TxBSxmarker_dim (if real)
                         Tensor of shape TxBSx1(if categorical)
                    t   : Tensor of shape TxBSx2. [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                    mask: Tensor of shape TxBS If mask[t,i] =1 then that timestamp is present
                Output:
                    loss : Tensor scalar
            """
            #TxBS and TxBS
            time_log_likelihood, marker_log_likelihood = self._forward(x, t, mask)
            loss = (time_log_likelihood + marker_log_likelihood) * mask
            loss. mean()

        def run_forward_rnn(self, x, t):
            """
                Input: 
                    x   : Tensor of shape TxBSxmarker_dim (if real)
                         Tensor of shape TxBSx1(if categorical)
                    t   : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                Output:
                    h : Tensor of shape (T+1)xBSxhidden_dim
                    embed_h : Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
            """
            batch_size , seq_length = x.size(1), x.size(0)
            #phi Tensor shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
            _, _, phi = self.preprocess_input(x, t)

            outs = []
            h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
            outs.append(h_t[None,:, :])
            for seq in range(seq_lengths):
                h_t = self.rnn_cell(phi[ seq,:, :], h_t)
                outs.append(h_t[None,:,:])
            
            h = torch.cat(outs, dim =0) # shape = [T+1, batchsize, h]
            return h, self.preprocess_hidden_state(h)

        def preprocess_hidden_state(self, h):
            return self.embed_hidden_state(h)


        def preprocess_input(self,x,t):
            """
                Input: 
                    x   : Tensor of shape TxBSxmarker_dim (if real)
                         Tensor of shape TxBSx1(if categorical)
                    t   : Tensor of shape TxBSx2. [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}

                Output:
                    phi_x : Tensor of shape TxBSx self.x_embedding_layer[-1]
                    phi_t : Tensor of shape TxBSx self.t_embedding_layer[-1]
                    phi   : Tensor of shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
            """
            if self.marker_type != real:
                x = one_hot_encoding(x)# Shape TxBSxmarker_dim
            phi_x = self.embed_x(x)
            phi_t = self.embed_t(t)
            phi   = torch.cat([phi_x, phi_t], -1)
            return phi_x, phi_t, phi


        def _forward(self, x, t, mask):
            """
                Input: 
                    x   : Tensor of shape TxBSxmarker_dim (if real)
                         Tensor of shape TxBSx1(if categorical)
                    t   : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                    mask: Tensor of shape TxBSx1. If mask[t,i,0] =1 then that timestamp is present
                Output:
                    
            """

            #Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
            _, hidden_states = self.run_forward_rnn(x, t)

            #marker generation layer. Ideally it should include time gap also.
            #Tensor of shape TxBSx marker_dim
            marker_out_mu, marker_out_logvar = self.generate_marker(hidden_states, t)

            time_log_likelihood = self.compute_point_log_likelihood(hidden_states, t)
            marker_log_likelihood = self.compute_marker_log_likelihood(x, marker_out_mu, marker_out_logvar)

            return time_log_likelihood, marker_log_likelihood #TxBS and TxBS

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
                x_recon_dist = Normal(mu, logvar.exp().sqrt())
                nll_loss = (-x_recon_dist.log_prob(x)).sum(dim = -1, keepdim = True)
                return nll_loss
            else:
                seq_lengths, batch_size = x.size(0), x.size(1)
                mu_ = mu.view(-1, self.marker_dim)# T*BS x marker_dim
                x_ = x.view(-1) #(T*BS,)
                loss = F.cross_entropy(mu_,x_, reduction= 'none').view(seq_lengths, batch_size)
                return -loss


        def compute_point_log_likelihood(hidden_states, t):
            """
                Input:
                    h : Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
                    t : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                Output:
                    log_f_t : tensor of shape TxBS
                    
            """
            h_trimmed = h[:-1,:,:]#TxBSxself.shared_output_layers[-1]
            d_js = t[:,:,1][:,:,None] #Shape TxBSx1 Time differences

            past_influence = self.output_time_dic['h_influence'](h_trimmed) #TxBSx1
            current_influence = self.output_time_dic['time_influence'] * d_js #TxBSx1
            base_intensity = self.output_time_dic['base_intensity'] #1x1x1

            term1 = past_influence + current_influence + base_intensity
            term2 = (past_influence+ base_intensity).exp()
            term3 = term1.exp()

            log_f_t = term1 + (1./self.output_time_dic['time_influence']) * (term2-term3)
            return log_f_t[:,:, 0]#TxBS



        def generate_marker(self, h, t):
            """
                Input:
                    h : Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
                    t : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                Output:
                    marker_out_mu : Tensor of shape T x BS x marker_dim
                    marker_out_logvar : Tensor of shape T x BS x marker_dim #None in case of non real marker
            """
            h_trimmed = h[:-1,:,:]
            if self.x_given_t:
                d_js = t[:,:,1][:,:,None] #Shape TxBSx1 Time differences
                h_trimmed = torch.cat([h_trimmed, d_js], -1)
            marker_out_mu = self.output_x_mu(h_trimmed)

            if self.marker_type == 'real':
                marker_out_logvar = self.output_x_logvar(h_trimmed)
            else:
                marker_out_logvar = None
            return marker_out_mu, marker_out_logvar













