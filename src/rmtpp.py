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


class rmtpp(nn.Module):
    """
        Implementation of Recurrent Marked Temporal Point Processes: Embedding Event History to Vector
        'https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf'

        ToDo:
            1. Mask not implemented
            2.

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
        self.embed_hidden_state, self.output_x , self.output_time_dic = self.create_output_layer()


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

            if self.x_given_t:
                x_module = nn.Linear(self.shared_output_layers[-1]+1, self.marker_dim)
            else:
                x_module = nn.Linear(self.shared_output_layers[-1], self.marker_dim)

            time_module_dic = nn.ModuleDict({
                'h_influence': nn.Linear(shared_output_layers[-1], 1, bias = False),
                'time_influence': nn.Parameter(torch.zeros(1,1)),
                'base_intensity': nn.Parameter(torch.zeros(1,1))
                })
            return embed_module, x_module, time_module_dic



        def forward(self, x, t, mask  = None):
            """
                Input: 
                    x   : Tensor of shape TxBSxmarker_dim
                    t   : Tensor of shape TxBSx2. [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                    mask: Tensor of shape TxBSx1. If mask[t,i,0] =1 then that timestamp is present
                Output:
                    loss : Tensor scalar
            """
            self._forward(x, t, mask)

        def run_forward_rnn(self, x, t):
            """
                Input: 
                    x   : Tensor of shape TxBSxmarker_dim
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
                    x   : Tensor of shape TxBSxmarker_dim
                    t   : Tensor of shape TxBSx2. [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}

                Output:
                    phi_x : Tensor of shape TxBSx self.x_embedding_layer[-1]
                    phi_t : Tensor of shape TxBSx self.t_embedding_layer[-1]
                    phi   : Tensor of shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
            """
            phi_x = self.embed_x(x)
            phi_t = self.embed_t(t)
            phi   = torch.cat([phi_x, phi_t], -1)
            return phi_x, phi_t, phi


        def _forward(self, x, t, mask):
            """
                Input: 
                    x   : Tensor of shape TxBSxmarker_dim
                    t   : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                    mask: Tensor of shape TxBSx1. If mask[t,i,0] =1 then that timestamp is present
                Output:
                    
            """

            #Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
            _, hidden_states = self.run_forward_rnn(x, t)

            #marker generation layer. Ideally it should include time gap also.
            #Tensor of shape TxBSx marker_dim
            marker_out = self.generate_marker(hidden_states, t)

            

        def generate_marker(self, h, t):
            """
                Input:
                    h : Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
                    t : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                        [i,:,1] represents time gap d_i = t_i- t_{i-1}
                Output:
                    marker_out : Tensor of shape T x BS x marker_dim
            """
            h_trimmed = h[:-1,:,:]
            if self.x_given_t:
                d_js = t[:,:,1][:,:,None] #Shape TxBSx1 Time differences
                h_trimmed = torch.cat([h_trimmed, d_js], -1)
            marker_out = self.output_x(h_trimmed)
            return marker_out













