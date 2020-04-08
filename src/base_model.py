import torch
from torch import Tensor
import numpy as np
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils
from torch.distributions import Normal, Categorical, Gumbel
from torch.distributions import kl_divergence
from torch.optim import Adam
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel_softmax(logits, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar

        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = Gumbel(torch.zeros(*logits.shape), torch.ones(*logits.shape)).sample().to(device)
    # g = sample_gumbel(logits.shape)
    # assert g.shape == logits.shape
    h = (g + logits) / temperature
    y = F.softmax(h, dim=-1)
    return y


class MLP(nn.Module):
    def __init__(self, dims: list):
        assert len(dims) >= 2  # should at least be [inputdim, outputdim]
        super().__init__()
        layers = list()
        for i in range(len(dims) - 1):
            n = dims[i]
            m = dims[i + 1]
            L = nn.Linear(n, m, bias=True)
            layers.append(L)
            layers.append(nn.ReLU())  # NOTE: Always slaps a non-linearity in the end
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPNormal(MLP):
    def __init__(self, dims: list):
        try:
            assert len(dims) >= 3  # should at least be [inputdim, hiddendim1, outdim]
        except AssertionError:
            print(dims)
            raise

        super().__init__(dims[:-1])  # initializes the core network
        self.mu_module = nn.Linear(dims[-2], dims[-1], bias=False)
        self.logvar_module = nn.Linear(dims[-2], dims[-1], bias=False)

    def forward(self, x):
        h = self.net(x)
        mu, logvar = self.mu_module(h), self.logvar_module(h)
        dist = Normal(mu, logvar.div(2).exp())  # std = exp(logvar/2)
        return dist


class MLPCategorical(MLP):
    def __init__(self, dims: list):
        try:
            assert len(
                dims) >= 3  # should at least be [inputdim, hiddendim1, logitsdim] - otherwise it's just a matrix multiplication
        except AssertionError:
            print(dims)
            raise

        super().__init__(dims[:-1])  # initializes the core network
        self.logit_module = nn.Linear(dims[-2], dims[-1], bias=False)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        logits = self.logit_module(h)
        dist = Categorical(logits=logits)
        return dist


class BaseEncoder(nn.Module):
    def __init__(self, rnn_dims: list, y_dims: list, z_dims: list):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.y_dims = y_dims
        self.z_dims = z_dims
        self.rnn_hidden_dim = rnn_dims[-1]
        self.y_dim = y_dims[-1]
        self.latent_dim = z_dims[-1]
        self.y_module, self.rnn_module, self.z_module = self.create_inference_nets()

    def create_inference_nets(self):
        y_module = MLPCategorical(self.y_dims)

        rnn = nn.GRU(
            input_size=self.rnn_dims[0],
            hidden_size=self.rnn_dims[1],
        )
        z_module = MLPNormal(self.z_dims)
        return y_module, rnn, z_module

    def forward(self, xt, temp, mask):
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self, shared_output_dims: list, marker_dim: int, decoder_in_dim: int, **kwargs):
        super().__init__()
        self.shared_output_dims = shared_output_dims
        self.marker_dim = marker_dim
        self.time_loss = kwargs['time_loss']
        self.marker_type = kwargs['marker_type']
        self.x_given_t = kwargs['x_given_t']
        self.preprocessing_module_dims = [decoder_in_dim, *self.shared_output_dims]
        self.preprocessing_module = self.create_generative_nets()

    def generate_marker(self, h, t):
        mu, logvar = generate_marker(self, h, t)
        if self.marker_type == 'real':
            return Normal(mu, logvar.div(2).exp())
        elif self.marker_type == 'categorical':
            return Categorical(logits=mu)
        else:
            raise NotImplementedError

    def create_generative_nets(self):
        gen_pre_module = MLP(self.preprocessing_module_dims)
        return gen_pre_module

    def compute_time_log_prob(self, h, t):
        return compute_point_log_likelihood(self, h, t)

    def compute_marker_log_prob(self, x, dist_x_recon: torch.distributions.Distribution):
        if dist_x_recon.__class__.__name__ == "Normal":
            return dist_x_recon.log_prob(x)
        elif dist_x_recon.__class__.__name__ == "Categorical":
            return dist_x_recon.log_prob(x)
        else:
            raise NotImplementedError

    def forward(self, concat_hzy):
        raise NotImplementedError


class BaseModel(nn.Module):
    model_name = ""

    def __init__(self, **kwargs):
        super().__init__()
        self.marker_type = kwargs['marker_type']
        self.marker_dim = kwargs['marker_dim']
        self.time_dim = kwargs['time_dim']
        self.rnn_hidden_dim = kwargs['rnn_hidden_dim']
        self.latent_dim = kwargs['latent_dim']
        self.cluster_dim = kwargs['n_cluster']
        self.x_given_t = kwargs['x_given_t']
        self.time_loss = kwargs['time_loss']
        self.base_intensity = kwargs['base_intensity']
        self.time_influence = kwargs['time_influence']

        ## Preprocessing networks
        # Embedding network
        self.x_embedding_dim = [128]
        self.t_embedding_dim = [8]
        self.emb_dim = self.x_embedding_dim[-1] + self.t_embedding_dim[-1]
        self.embed_x, self.embed_t = self.create_embedding_nets()
        self.shared_output_dims = [256]

        # Inference network
        if self.latent_dim is not None:
            self.encoder_z_hidden_dims = [64, 64]
            self.encoder_y_hidden_dims = [64]
            z_input_dim = self.rnn_hidden_dim + self.emb_dim + self.cluster_dim
            self.rnn_dims = [self.emb_dim, self.rnn_hidden_dim]
            self.y_dims = [self.rnn_hidden_dim, *self.encoder_y_hidden_dims, self.cluster_dim]
            self.z_dims = [z_input_dim, *self.encoder_z_hidden_dims, self.latent_dim]
        # self.encoder = Encoder(rnn_dims=rnn_dims, y_dims=y_dims, z_dims=z_dims)

    def print_parameter_info(self):
        # dump whatever str(nn.Module) has to offer
        print(self)

        # dump parameter info
        for pname, pdata in self.named_parameters():
            print(f"{pname}: {pdata.size()}")

    def create_embedding_nets(self):
        # marker_dim is passed. timeseries_dim is 2
        if self.marker_type == 'categorical':
            x_module = nn.Embedding(self.marker_dim, self.x_embedding_dim[0])
        else:
            x_module = nn.Sequential(
                nn.Linear(self.marker_dim, self.x_embedding_dim[0]),
                nn.ReLU(),
            )
            # raise NotImplementedError

        t_module = nn.Sequential(
            nn.Linear(self.time_dim, self.t_embedding_dim[0]),
            nn.ReLU()
        )
        return x_module, t_module

    def compute_metrics(self, marker_logits, predicted_times, marker, event_times, mask):
        """
        Input:
            marker_logits : Tensor of shape T x BS x marker_dim; t_j : actual time of event j
            predicted_times : Tensor of shape T x BS x 1 ; pt_j : predicted time of event j
            marker : Tensor of shape T x BS
            event_times : Tensor of shape T x BS x 1
            mask: Tensor of shape T x BS
        Output:
            metric_dict: dict
        """
        metric_dict = {}
        with torch.no_grad():
            # note: both t[0] and pt[0] are 0 by convention
            time_mse = torch.pow((predicted_times - event_times) * mask.unsqueeze(-1), 2.)  # (T, BS, 1)
            metric_dict['time_mse'] = time_mse.sum().detach().cpu().numpy()
            # note: because 0th timesteps are zero always, reducing count to ensure accuracy stays unbiased
            metric_dict['time_mse_count'] = mask[1:, :].sum().detach().cpu().numpy()

            if self.marker_type == "categorical":
                predicted = torch.argmax(marker_logits, dim=-1)  # (T, BS)
                correct_predictions = (predicted == marker) * mask  # (T, BS)
                correct_predictions = correct_predictions[1:]  # Keep only the predictions from 2nd timestep

                metric_dict[
                    'marker_acc'] = correct_predictions.sum().detach().cpu().numpy()  # count how many correct predictions we made
                metric_dict['marker_acc_count'] = (
                    mask[1:, :]).sum().cpu().numpy()  # count how many predictions we made
            else:
                raise NotImplementedError

        return metric_dict


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


def assert_input(self):
    assert self.marker_type in {
        'real', 'categorical', 'binary'}, "Unknown Input type provided!"


def create_input_embedding_layer(model):
    if model.marker_type == 'categorical':
        x_module = nn.Embedding(model.marker_dim, model.x_embedding_dim[0])
    else:
        x_module = nn.Sequential(
            nn.Linear(model.marker_dim, model.x_embedding_dim[0])  # , nn.ReLU(),
            # Not sure whether to put Relu at the end of embedding layer
            # nn.Linear(self.x_embedding_dim[0],
            #          self.x_embedding_dim[1]), nn.ReLU()
        )

    # t_module = nn.Linear(self.time_dim, self.t_embedding_dim[0])
    t_module = nn.Sequential(
        nn.Linear(model.time_dim, model.t_embedding_dim[0])  # ,
        # nn.ReLU(),
        # nn.Dropout(p=0.5),
        # nn.Linear(model.t_embedding_dim[0], model.t_embedding_dim[0])
    )
    return x_module, t_module


def create_output_marker_layer(model):
    embed_module = nn.Sequential(
        nn.ReLU(), nn.Dropout(model.dropout),
        nn.Linear(model.hidden_embed_input_dim, model.shared_output_dims[0]),
        nn.ReLU(), nn.Dropout(model.dropout)
        # nn.Linear(
        #    self.shared_output_dims[0], self.shared_output_dims[1]), nn.ReLU()
    )

    x_module_logvar = None
    l = model.shared_output_dims[-1]
    if model.x_given_t:
        l += 1
    if model.marker_type == 'real':
        x_module_mu = nn.Linear(l, model.marker_dim)
        x_module_logvar = nn.Linear(l, model.marker_dim)
    elif model.marker_type == 'binary':  # Fix binary
        x_module_mu = nn.Sequential(
            nn.Linear(l, model.marker_dim),
            nn.Sigmoid())
    elif model.marker_type == 'categorical':
        x_module_mu = nn.Sequential(
            nn.Linear(l, model.marker_dim)  # ,
            # nn.Softmax(dim=-1)
        )

    return embed_module, x_module_mu, x_module_logvar


# def create_output_time_layer(model, b, ti):
#     l =model.shared_output_dims[-1]
#     if model.time_loss == 'intensity':
#         h_influence =  nn.Linear(l, 1, bias=False)
#         time_influence = nn.Parameter(ti*torch.ones(1, 1, 1))#0.005*
#         base_intensity =  nn.Parameter(torch.zeros(1, 1, 1)-b)#-8
#         model.h_influence, model.time_influence, model.base_intensity =  h_influence, time_influence, base_intensity
#     else:
#         model.time_mu =   nn.Linear(l, 1)
#         model.time_logvar =   nn.Linear(l, 1)
#     return

def create_output_nets(model, b, ti):
    """
    b: (float) base intensity #TODO
    ti: (float) time influence #TODO
    """

    l = model.shared_output_dims[-1]

    # Output net for time
    if model.time_loss == 'intensity':
        h_influence = nn.Linear(l, 1, bias=False)
        time_influence = nn.Parameter(ti * torch.ones(1, 1, 1))  # 0.005*
        base_intensity = nn.Parameter(torch.zeros(1, 1, 1) - b)  # -8
        model.h_influence, model.time_influence, model.base_intensity = h_influence, time_influence, base_intensity
    else:
        model.time_mu = nn.Linear(l, 1)
        model.time_logvar = nn.Linear(l, 1)

    # Output net for markers
    x_module_logvar = None
    if model.x_given_t:
        l += 1
    if model.marker_type == 'real':
        x_module_mu = nn.Linear(l, model.marker_dim)
        x_module_logvar = nn.Linear(l, model.marker_dim)
    elif model.marker_type == 'binary':  # Fix binary
        x_module_mu = nn.Sequential(
            nn.Linear(l, model.marker_dim),
            nn.Sigmoid())
    elif model.marker_type == 'categorical':
        x_module_mu = nn.Sequential(
            nn.Linear(l, model.marker_dim)  # ,
            # nn.Softmax(dim=-1)
        )
    model.output_x_mu, model.output_x_logvar = x_module_mu, x_module_logvar


def compute_marker_log_likelihood(model, x, mu, logvar):
    """
        Input:  
                x   : Tensor of shape TxBSxmarker_dim (if real)
                    Tensor of shape TxBSx1(if categorical)
                mu : Tensor of shape T x BS x marker_dim
                logvar : Tensor of shape T x BS x marker_dim #None in case of non real marker
        Output:
                loss : TxBS
    """
    if model.marker_type == 'real':
        sigma = torch.clamp(logvar.exp().sqrt(), min=model.sigma_min)
        x_recon_dist = Normal(mu, sigma)
        ll_loss = (x_recon_dist.log_prob(x)
                   ).sum(dim=-1)
        return ll_loss
    else:
        seq_lengths, batch_size = x.size(0), x.size(1)

        if model.marker_type == 'categorical':
            mu_ = mu.view(-1, model.marker_dim)  # T*BS x marker_dim
            x_ = x.view(-1)  # (T*BS,)
            loss = F.cross_entropy(mu_, x_, reduction='none').view(
                seq_lengths, batch_size)
        else:  # binary
            loss = F.binary_cross_entropy(mu, x, reduction='none').sum(dim=-1)  # TxBS
        return -loss

def compute_point_log_likelihood(model, h, t):
    """
        Input:
            h : Tensor of shape (T)xBSxself.shared_output_dims[-1]
            t : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
                [i,:,1] represents time gap d_i = t_i- t_{i-1}
        Output:
            log_f_t : tensor of shape TxBS

    """
    h_trimmed = h  # TxBSxself.shared_output_dims[-1]
    d_js = t[:, :, 0][:, :, None]  # Shape TxBSx1 Time differences

    if model.time_loss == 'intensity':
        past_influence = model.h_influence(h_trimmed)  # TxBSx1

        # TxBSx1
        if model.time_influence > 0:
            ti = torch.clamp(model.time_influence, min=1e-5)
        else:
            ti = torch.clamp(model.time_influence, max=-1e-5)
        current_influence = ti * d_js
        base_intensity = model.base_intensity  # 1x1x1

        term1 = past_influence + current_influence + base_intensity
        term2 = (past_influence + base_intensity).exp()
        term3 = term1.exp()

        log_f_t = term1 + \
                  (1. / (ti)) * (term2 - term3)
        return log_f_t[:, :, 0], None  # TxBS
    else:
        mu_time = model.time_mu(h_trimmed)  # TxBSx1
        logvar_time = model.time_logvar(h_trimmed)  # TxBSx1
        sigma_time = logvar_time.exp().sqrt() + model.sigma_min  # TxBSx1
        time_recon_dist = Normal(mu_time, sigma_time)
        ll_loss = (time_recon_dist.log_prob(d_js)
                   ).sum(dim=-1)  # TxBS
        return ll_loss, mu_time


def preprocess_input(model, x, t):
    """
        Input: 
            x   : Tensor of shape TxBSxmarker_dim (if real)
                    Tensor of shape TxBSx1(if categorical)
            t   : Tensor of shape TxBSxtime_dim. [i,:,0] represents actual time at timestep i ,\
                [i,:,1] represents time gap d_i = t_i- t_{i-1}

        Output:
            phi_x : Tensor of shape TxBSx self.x_embedding_dim[-1]
            phi_t : Tensor of shape TxBSx self.t_embedding_dim[-1]
            phi   : Tensor of shape TxBS x (self.x_embedding_dim[-1] + self.t_embedding_dim[-1])
    """
    # if model.marker_type == 'categorical':
    #     # Shape TxBSxmarker_dim
    #     x = one_hot_encoding(x[:, :], model.marker_dim).to(device)
    phi_x = model.embed_x(x)
    phi_t = model.embed_time(t)
    # phi_t = t
    phi = torch.cat([phi_x, phi_t], -1)
    return phi_x, phi_t, phi


def generate_marker(model, h, t):
    """
        Input:
            h : Tensor of shape TxBSxself.shared_output_dims[-1]
            t : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
                [i,:,1] represents time gap d_i = t_i- t_{i-1}
        Output:
            marker_out_mu : Tensor of shape T x BS x marker_dim
            marker_out_logvar : Tensor of shape T x BS x marker_dim #None in case of non real marker
    """
    # h_trimmed = h[:-1, :, :]
    h_trimmed = h
    if model.x_given_t:
        d_js = t[:, :, 1][:, :, None]  # Shape TxBSx1 Time differences
        h_trimmed = torch.cat([h_trimmed, d_js], -1)

    marker_out_mu = model.output_x_mu(h_trimmed)

    if model.marker_type == 'real':
        marker_out_logvar = model.output_x_logvar(h_trimmed)
    else:
        marker_out_logvar = None
    return marker_out_mu, marker_out_logvar
