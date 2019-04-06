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
# import pdb; pdb.set_trace()
from base_model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.metric import get_marker_metric, compute_time_expectation
DEBUG = False



class hrmtpp(nn.Module):
    """
        Implementation of Proposed Hierarchichal Recurrent Marked Temporal Point Processes
        ToDo:
            1. Mask verify
            2. for categorical, x is TxBSx1. create embedding layer with one hot vector.
            3. time parameter is still a simple linear function of past present and base intensity

    """

    def __init__(self, latent_dim = 20, marker_type='real', marker_dim=31, time_dim=2, hidden_dim=128, x_given_t=False,base_intensity = 0.,time_influence = 1., gamma = 1., time_loss = 'intensity'):
        super().__init__()
        """
            Input:
                marker_type : 'real' or 'binary', 'categorical'
                marker_dim  : number of dimension  in case of real input. Otherwise number of classes for categorical
                but marker_dim is 1 (just the class label)
                hidden_dim : hidden dimension is gru cell
                latent_dim : sub-population vector
                x_given_t : whether to condition marker given time gap. For RMTPP set it false.
        """

        self.marker_type = marker_type
        self.marker_dim = marker_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.x_given_t = x_given_t
        self.use_rnn_cell = False
        self.time_loss = time_loss
        self.gamma = gamma
        self.assert_input()

        self.logvar_min = np.log(1e-4)
        self.sigma_min = 1e-2

        # Set up layer dimensions. This is only hidden layers dimensions
        self.x_embedding_layer = [64]
        self.t_embedding_layer = [64]
        self.shared_output_layers = [64]
        self.encoder_layers = [64, 64]

        self.hidden_embed_input_dim = self.hidden_dim + self.latent_dim

        # setup layers
        self.embed_x, self.embed_time = create_input_embedding_layer(self)
        self.forward_rnn_cell, self.backward_rnn_cell = self.create_rnn_network()
        self.inference_net, self.inference_mu, self.inference_logvar = self.create_inference_network()
        self.embed_hidden_state, self.output_x_mu, self.output_x_logvar = create_output_marker_layer(self)
        create_output_time_layer(self, base_intensity, time_influence)

    def assert_input(self):
        assert self.marker_type in {
            'real', 'categorical', 'binary'}, "Unknown Input type provided!"

    def create_rnn_network(self):
        if self.use_rnn_cell:
            rnn   = nn.GRUCell(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            rnn_b = nn.GRUCell(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            return rnn , rnn_b
        else:
            rnn   = nn.GRU(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            rnn_b = nn.GRU(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            return rnn , rnn_b

    def create_inference_network(self):
        encoder = nn.Sequential(
            nn.Linear(2* self.hidden_dim,self.encoder_layers[-1]), nn.ReLU()#,
            #nn.Linear(self.encoder_layers[0], self.encoder_layers[1]), nn.ReLU()
        )
        inference_mu = nn.Linear(self.encoder_layers[-1], self.latent_dim)
        inference_logvar = nn.Linear(self.encoder_layers[-1], self.latent_dim)
        return encoder, inference_mu, inference_logvar


    def forward(self, x, t, anneal = 1., mask=None):
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
        time_log_likelihood, marker_log_likelihood, kl_loss, metric_dict = self._forward(x, t, mask)

        marker_loss = (-1.* marker_log_likelihood *mask).sum()
        time_loss = (-1. *time_log_likelihood *mask).sum()
        kl_loss = kl_loss.sum()

        loss = self.gamma*time_loss + marker_loss + anneal* kl_loss
        true_loss = time_loss + marker_loss +kl_loss
        meta_info = {"marker_ll":marker_loss.detach().cpu(), "time_ll":time_loss.detach().cpu(), "kl_loss":kl_loss.detach().cpu(), "true_ll": true_loss.detach().cpu()}
        return loss, {**meta_info, **metric_dict}

    def run_forward_backward_rnn(self, x, t, mask):
        """
            Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
                mask: Tensor of shape TxBSx If mask[t,i] =1 then that timestamp is present
            Output:
                h : Tensor of shape (T)xBSxhidden_dim
                h_b : Tensor of shape (T)xBSxhidden_dim
        """
        batch_size, seq_length = x.size(1), x.size(0)
        # phi Tensor shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
        _, _, phi = preprocess_input(self, x, t)


        if self.use_rnn_cell:
            outs = []
            h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
            outs.append(h_t[None, :, :])
            for seq in range(seq_length):
                h_t = self.forward_rnn_cell(phi[seq, :, :], h_t)
                if mask is None:
                    outs.append(h_t[None, :, :])
                else:
                    outs.append(h_t[None, :, :] * mask[seq,:][None,:,None])# 1xBSxhidden_dim * 1xBSx1 Broadcasting

            h = torch.cat(outs, dim=0)  # shape = [T+1, batchsize, h]

            phi_flipped = torch.flip(phi, [0])
            if mask is not None:
                mask_flipped = torch.flip(mask, [0])
            reverse_outs = []
            rh_t = torch.zeros(batch_size, self.hidden_dim).to(device)
            reverse_outs.append(rh_t[None,:,:])
            for seq in range(seq_length):
                rh_t = self.backward_rnn_cell(phi_flipped[seq, :, :], rh_t)
                if mask is None:
                    reverse_outs.append(rh_t[None, :, :])
                else:
                    reverse_outs.append(rh_t[None, :, :] * mask_flipped[seq,:][None,:,None])# 1xBSxhidden_dim * 1xBSx1 Broadcasting

            rh_flipped = torch.cat(reverse_outs, dim=0)  # shape = [T+1, batchsize, h]
            rh = torch.flip(rh_flipped, [0])

            #Now for generation at time i we need h_{i-1}. For encoder network at time i, we need a_i in the reverse rnn
            return h[:-1,:,:], rh[1:,:,:]
        else:
            # Run RNN over the concatenated sequence [marker_seq_emb, time_seq_emb]
            h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
            hidden_seq, _ = self.forward_rnn_cell(phi, h_0)
            h = torch.cat([h_0, hidden_seq], dim = 0)[:-1,:,:]

            phi_flipped = torch.flip(phi, [0])
            rh_0 = torch.zeros(1,batch_size, self.hidden_dim).to(device)
            r_hidden_seq, _ = self.backward_rnn_cell(phi_flipped, rh_0)
            rh_flipped = torch.cat([rh_0, r_hidden_seq], dim = 0)
            rh = torch.flip(rh_flipped, [0])[1:,:,:]

            return h, rh

    def compute_hidden_states(self, x, t, mask):
        """
        Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSxtime_dim. [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
        Output:
                hz : Tensor of shape (T)xBSxself.shared_output_layers[-1]
        """
        hs, back_hs = self.run_forward_backward_rnn(x, t, mask)

        mu, logvar = self.encoder(hs, back_hs) #TxBSxlatent_dim
        z = self.reparameterize(mu[0,:,:], logvar[0,:,:])[None,:,:]#of shape 1xBSxlatent_dim

        hz_embedded = self.preprocess_hidden_latent_state(hs, z)
        return hz_embedded

    def preprocess_hidden_latent_state(self, h, z):
        """
            Input:
                h : (T)xBSxhidden_dim
                z : 1xBSxlatent_dim

            output:
                out: TxBSxshared_output_layers[-1]
        """
        T = h.size(0)
        repeat_vals = (T, -1, -1)
        z_broadcast = z.expand(*repeat_vals)

        hz = torch.cat([h, z_broadcast], dim = -1)
        return self.embed_hidden_state(hz)


    def encoder(self, hs, back_hs):
        """
            We do not need to compute mu of shape TxBSxLatent_dim. In case of static embedding only the first one will
            be used. But any of them can be a unbiased estimator. Need to add all of them with penalty that they are from
            same distribution
            input:
                h : Tensor of shape (T)xBSxhidden_dim
                h_b : Tensor of shape (T)xBSxhidden_dim
            output:
                mu : TxBSxlatent_dim
                log_var: TxBSxlatent_dim

        """
        hiddenlayer = self.inference_net(torch.cat([hs, back_hs], -1))
        mu = self.inference_mu(hiddenlayer)
        logvar = torch.clamp(self.inference_logvar(hiddenlayer), min = self.logvar_min)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        return mu + epsilon.mul(sigma)

    def _forward(self, x, t, mask):
        """
            Input: 
                x   : Tensor of shape TxBSxmarker_dim (if real)
                     Tensor of shape TxBSx1(if categorical)
                t   : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
                mask: Tensor of shape TxBS. If mask[t,i] =1 then that timestamp is present
            Output:

        """

        # Tensor of shape (T)xBSxhidden_dim
        hs, back_hs = self.run_forward_backward_rnn(x, t, mask)

        mu, logvar = self.encoder(hs, back_hs) #TxBSxlatent_dim
        z = self.reparameterize(mu[0,:,:], logvar[0,:,:])[None,:,:]#of shape 1xBSxlatent_dim

        
        hz_embedded = self.preprocess_hidden_latent_state(hs, z)
        # marker generation layer. Ideally it should include time gap also.
        # Tensor of shape TxBSx marker_dim
        marker_out_mu, marker_out_logvar = generate_marker(self, hz_embedded,  t)
        marker_log_likelihood = compute_marker_log_likelihood(self, x, marker_out_mu, marker_out_logvar)

        
        time_log_likelihood, mu_time = compute_point_log_likelihood(self, hz_embedded,  t)
        metric_dict = {}
        with torch.no_grad():
            get_marker_metric(self.marker_type, marker_out_mu, x, mask, metric_dict)
            if self.time_loss == 'intensity':
                expected_t = compute_time_expectation(self, hz_embedded, t, mask)
                time_mse = torch.abs(expected_t- t[:,:,0])[1:, :] * mask[1:, :]
            else:
                time_mse = torch.abs(mu_time[:,:,0]- t[:,:,0])[1:, :] * mask[1:, :]
            metric_dict['time_mse'] = time_mse.sum().detach().cpu().numpy()
            metric_dict['time_mse_count'] = mask[1:,:].sum().detach().cpu().numpy()

        posterior_dist = Normal(mu[0,:,:], logvar[0,:,:].exp().sqrt())
        prior_dist = Normal(0, 1)
        kld_loss = kl_divergence(posterior_dist, prior_dist)#Shape BSxlatent_dim

        return time_log_likelihood, marker_log_likelihood, kld_loss, metric_dict  # TxBS and TxBS

if __name__ == "__main__":
    model = hrmtpp()
