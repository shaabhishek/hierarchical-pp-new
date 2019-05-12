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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    return gumbel_softmax_sample(logits, temperature)



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


class h_storn_softmax(nn.Module):
    """
        Implementation of Proposed Hierarchichal Recurrent Marked Temporal Point Processes
        ToDo:
            1. Mask verify
            2. for categorical, x is TxBSx1. create embedding layer with one hot vector.
            3. time parameter is still a simple linear function of past present and base intensity

    """

    def __init__(self, marker_type='real', marker_dim=20, n_cluster=5, latent_dim=20, hidden_dim=128, x_given_t=False):
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
        self.n_cluster = n_cluster
        self.x_given_t = x_given_t
        self.use_rnn_cell = False
        self.assert_input()

        self.logvar_min = np.log(1e-4)

        # Set up layer dimensions. This is only hidden layers dimensions
        self.x_embedding_layer = [64]
        self.t_embedding_layer = [64]
        self.shared_output_layers = [64]
        self.encoder_layers = [64, 64]

        # setup layers
        self.embed_x, self.embed_time = self.create_input_embedding_layer()
        self.forward_rnn_cell, self.backward_rnn_cell = self.create_rnn_network()
        self.inference_net, self.inference_mu, self.inference_logvar = self.create_inference_network()
        self.embed_hidden_state, self.output_x_mu, self.output_x_logvar = self.create_output_marker_layer()
        self.h_influence, self.time_influence, self.base_intensity = self.create_output_time_layer()

        # Hierarchichal Layer
        self.cluster_rnn, self.inference_cluster, self.cluster_decoder, self.output_z_mu, self.output_z_logvar = self.create_cluster_layer()

    def create_cluster_layer(self):
        cluster_encoder = nn.GRU(
            input_size=self.latent_dim, hidden_size=self.latent_dim, bidirectional= True)
        inference_cluster = nn.Linear(self.latent_dim*2, self.n_cluster)
        cluster_decoder = encoder = nn.Sequential(
            nn.Linear(self.n_cluster, 100), nn.ReLU()
        )
        mu_z = nn.Linear(100, self.latent_dim)
        logvar_z = nn.Linear(100, self.latent_dim)

        return cluster_encoder, inference_cluster, cluster_decoder, mu_z, logvar_z

    def assert_input(self):
        assert self.marker_type in {
            'real', 'categorical', 'binary'}, "Unknown Input type provided!"

    def create_rnn_network(self):
        if self.use_rnn_cell:
            rnn = nn.GRUCell(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            rnn_b = nn.GRUCell(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            return rnn, rnn_b
        else:
            rnn = nn.GRU(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            rnn_b = nn.GRU(
                input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
            return rnn, rnn_b

    def create_inference_network(self):
        encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.x_embedding_layer[-1] +
                      self.t_embedding_layer[-1], self.encoder_layers[-1]), nn.ReLU()  # ,
            #nn.Linear(self.encoder_layers[0], self.encoder_layers[1]), nn.ReLU()
        )
        inference_mu = nn.Linear(self.encoder_layers[-1], self.latent_dim)
        inference_logvar = nn.Linear(self.encoder_layers[-1], self.latent_dim)

        return encoder, inference_mu, inference_logvar

    def create_input_embedding_layer(self):
        x_module = nn.Sequential(
            # , nn.ReLU(),
            nn.Linear(self.marker_dim, self.x_embedding_layer[0])
            # Not sure whether to put Relu at the end of embedding layer
            #nn.Linear(self.x_embedding_layer[0],self.x_embedding_layer[1]), nn.ReLU()
        )

        t_module = nn.Sequential(
            nn.Linear(2, self.t_embedding_layer[0]),
            nn.ReLU(),
            nn.Linear(self.t_embedding_layer[0], self.t_embedding_layer[0])
        )
        return x_module, t_module

    def create_output_marker_layer(self):
        embed_module = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim,
                      self.shared_output_layers[0])  # , nn.ReLU(),
            #nn.Linear(self.shared_output_layers[0], self.shared_output_layers[1]), nn.ReLU()
        )

        x_module_logvar = None
        l = self.shared_output_layers[-1]
        if self.x_given_t:
            l += 1
        x_module_mu = nn.Linear(l, self.marker_dim)
        if self.marker_type == 'real':
            x_module_logvar = nn.Linear(l, self.marker_dim)

        return embed_module, x_module_mu, x_module_logvar

    def create_output_time_layer(self):

        h_influence = nn.Linear(self.shared_output_layers[-1], 1, bias=False)
        time_influence = nn.Parameter(0.01*torch.ones(1, 1, 1))
        base_intensity = nn.Parameter(torch.zeros(1, 1, 1))
        return h_influence, time_influence, base_intensity

    def forward(self, x, t, anneal=1.,temp =1., mask=None):
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
        time_log_likelihood, marker_log_likelihood, kl_loss = self._forward(
            x, t, temp, mask)

        if DEBUG:
            print("Losses:", -time_log_likelihood.sum().item(),  -
                  marker_log_likelihood.sum().item(), kl_loss.sum().item())
        loss = -1. * (time_log_likelihood + marker_log_likelihood)
        if mask is not None:
            loss = loss * mask
        return loss.sum() + kl_loss.sum(), [-marker_log_likelihood.sum().item(), -time_log_likelihood.sum().item(), kl_loss.sum().item()]

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
        _, _, phi = self.preprocess_input(x, t)

        if self.use_rnn_cell:
            outs = []
            recog_outs = []
            h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
            rh_t = torch.zeros(batch_size, self.hidden_dim).to(device)
            outs.append(h_t[None, :, :])
            recog_outs.append(rh_t[None, :, :])

            for seq in range(seq_length):
                h_t = self.forward_rnn_cell(phi[seq, :, :], h_t)
                rh_t = self.backward_rnn_cell(phi[seq, :, :], rh_t)
                if mask is None:
                    outs.append(h_t[None, :, :])
                    recog_outs.append(rh_t[None, :, :])
                else:
                    # 1xBSxhidden_dim * 1xBSx1 Broadcasting
                    outs.append(h_t[None, :, :] * mask[seq, :][None, :, None])
                    # 1xBSxhidden_dim * 1xBSx1 Broadcasting
                    recog_outs.append(
                        rh_t[None, :, :] * mask[seq, :][None, :, None])

            h = torch.cat(outs, dim=0)  # shape = [T+1, batchsize, h]
            rh = torch.cat(recog_outs, dim=0)  # shape = [T+1, batchsize, h]

            # Now for generation at time i we need h_{i-1}. For encoder network at time i, we need a_i in the reverse rnn
            return h[:-1, :, :], rh[1:, :, :], phi
        else:
            # Run RNN over the concatenated sequence [marker_seq_emb, time_seq_emb]
            h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
            hidden_seq, _ = self.forward_rnn_cell(phi, h_0)
            h = torch.cat([h_0, hidden_seq], dim=0)

            rh_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
            r_hidden_seq, _ = self.backward_rnn_cell(phi, rh_0)
            rh = torch.cat([rh_0, r_hidden_seq], dim=0)

            return h[:-1, :, :], rh[1:, :, :], phi

    def preprocess_hidden_latent_state(self, h, z):
        """
            Input:
                h : (T)xBSxhidden_dim
                z : TxBSxlatent_dim

            output:
                out: TxBSxshared_output_layers[-1]
        """
        hz = torch.cat([h, z], dim=-1)
        return self.embed_hidden_state(hz)

    def preprocess_input(self, x, t):
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
        if self.marker_type == 'categorical':
            # Shape TxBSxmarker_dim
            x = one_hot_encoding(x[:, :, 0], self.marker_dim)
        phi_x = self.embed_x(x)
        phi_t = self.embed_time(t)
        phi = torch.cat([phi_x, phi_t], -1)
        return phi_x, phi_t, phi

    def encoder(self, hs, phi):
        """
            We do not need to compute mu of shape TxBSxLatent_dim. In case of static embedding only the first one will
            be used. But any of them can be a unbiased estimator. Need to add all of them with penalty that they are from
            same distribution
            input:
                h : Tensor of shape (T)xBSxhidden_dim
                phi : Tensor of shape (T)xBSx self.x_embedding_layer[-1] + self.t_embedding_layer[-1]
            output:
                mu : TxBSxlatent_dim
                log_var: TxBSxlatent_dim

        """
        hiddenlayer = self.inference_net(torch.cat([hs, phi], -1))
        mu = self.inference_mu(hiddenlayer)
        logvar = torch.clamp(self.inference_logvar(
            hiddenlayer), min=self.logvar_min)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        return mu + epsilon.mul(sigma)

    def encoder_cluster_layer (self, z):#z dim is TxBSxlatent_dim
        batch_size = z.size(1)
        h_0 = torch.zeros(2, batch_size, self.latent_dim).to(device)
        hidden_seq, h = self.cluster_rnn(z, h_0)#h is 2xBSxlatent_dim
        h = torch.cat ([h[0,:,:],h[1,:,:]], dim =-1) #BSx 2latent_dim

        y = self.inference_cluster(h) # BSx K
        return y
        #self.inference_cluster, self.clsuter_decoder, self.output_z_mu, self.output_z_logvar 
        
    def decoder_cluster_layer(self,z , y):
        """
            z: TxBSxhidden_dim
            y: BSxC
        """
        T = z.size(0)
        hidden_state = self.cluster_decoder(y[None,:,:]) #1xBSx100
        repeat_vals = (T, -1,-1)
        hidden_state = hidden_state.expand(*repeat_vals)#TxBSx100
        out_mu_z = self.output_z_mu(hidden_state) #TxBSx latent_dim
        out_logvar_z = self.output_z_logvar(hidden_state) #TxBSx Latent_dim

        sigma = out_logvar_z.exp().sqrt()
        z_recon_dist = Normal(out_mu_z, sigma)
        ll_loss = (z_recon_dist.log_prob(z)).sum(dim=-1)
        return ll_loss#TxBS

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
        hs, recog_hs, phi = self.run_forward_backward_rnn(x, t, mask)

        mu, logvar = self.encoder(recog_hs, phi)  # TxBSxlatent_dim
        z = self.reparameterize(mu, logvar)  # of shape TxBSxlatent_dim


        hz_embedded = self.preprocess_hidden_latent_state(hs, z)
        return hz_embedded

    def _forward(self, x, t, temp, mask):
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
        hs, recog_hs, phi = self.run_forward_backward_rnn(x, t, mask)

        mu, logvar = self.encoder(recog_hs, phi)  # TxBSxlatent_dim
        z = self.reparameterize(mu, logvar)  # of shape TxBSxlatent_dim

        #Hierarchichal Layer
        logits = self.encoder_cluster_layer(z) #BSxcategorical_dim
        y  = gumbel_softmax(logits, temp)#BSxcategorical_dim

        #Decoder layer
        z_log_likelihood = self.decoder_cluster_layer(z, y) #TxBSxC
        #KL divergence Loss
        prob_ = F.softmax(logits, dim =-1)
        log_ratio = torch.log(prob_ * self.n_cluster + 1e-20)#TxBSxC
        kl_cluster = torch.sum(prob_ * log_ratio, dim=-1)#TxBS


        hz_embedded = self.preprocess_hidden_latent_state(hs, z)
        time_log_likelihood = self.compute_point_log_likelihood(
            hz_embedded,  t)
        # marker generation layer. Ideally it should include time gap also.
        # Tensor of shape TxBSx marker_dim
        marker_out_mu, marker_out_logvar = self.generate_marker(
            hz_embedded,  t)
        marker_log_likelihood = self.compute_marker_log_likelihood(
            x, marker_out_mu, marker_out_logvar)

        posterior_dist = Normal(mu, logvar.exp().sqrt())
        prior_dist = Normal(0, 1)
        # Shape TxBSxlatent_dim
        kl_z = kl_divergence(posterior_dist, prior_dist).sum(dim =-1)

        # TxBS and TxBS, TxBS
        return time_log_likelihood, marker_log_likelihood, kl_cluster+kl_z

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
            sigma = logvar.exp().sqrt()
            x_recon_dist = Normal(mu, sigma)
            ll_loss = (x_recon_dist.log_prob(x)
                       ).sum(dim=-1)
            return ll_loss
        else:
            seq_lengths, batch_size = x.size(0), x.size(1)
            if self.type == 'categorical':
                mu_ = mu.view(-1, self.marker_dim)  # T*BS x marker_dim
                x_ = x.view(-1)  # (T*BS,)
                loss = F.cross_entropy(mu_, x_, reduction='none').view(
                    seq_lengths, batch_size)
            else:
                loss = F.binary_cross_entropy_with_logits(mu, x, reduction= 'none').sum(dim =-1)#TxBS
            return -loss

    def compute_point_log_likelihood(self, h, t):
        """
            Input:
                h : Tensor of shape TxBSxself.shared_output_layers[-1]
                t : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
            Output:
                log_f_t : tensor of shape TxBS

        """
        d_js = t[:, :, 1][:, :, None]  # Shape TxBSx1 Time differences

        past_influence = self.h_influence(h)  # TxBSx1

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
                h : Tensor of shape (T)xBSxself.shared_output_layers[-1]
                t : Tensor of shape TxBSx2 [i,:,0] represents actual time at timestep i ,\
                    [i,:,1] represents time gap d_i = t_i- t_{i-1}
            Output:
                marker_out_mu : Tensor of shape T x BS x marker_dim
                marker_out_logvar : Tensor of shape T x BS x marker_dim #None in case of non real marker
        """
        h_trimmed = h
        if self.x_given_t:
            d_js = t[:, :, 1][:, :, None]  # Shape TxBSx1 Time differences
            h_trimmed = torch.cat([h_trimmed, d_js], -1)
        marker_out_mu = self.output_x_mu(h_trimmed)

        if self.marker_type == 'real':
            marker_out_logvar = torch.clamp(
                self.output_x_logvar(h_trimmed), min=self.logvar_min)
        else:
            marker_out_logvar = None
        return marker_out_mu, marker_out_logvar


if __name__ == "__main__":
    model = h_storn_softmax()
