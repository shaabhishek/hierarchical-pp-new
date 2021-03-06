import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_input_embedding_layer(model):
    if model.marker_type == 'categorical':
        x_module = nn.Embedding(model.marker_dim, model.x_embedding_layer[0])
    else:
        x_module = nn.Sequential(
            nn.Linear(model.marker_dim, model.x_embedding_layer[0]), nn.ReLU(),
            # Not sure whether to put Relu at the end of embedding layer
            # nn.Linear(self.x_embedding_layer[0],
            #          self.x_embedding_layer[1]), nn.ReLU()
        )

    # t_module = nn.Linear(self.time_dim, self.t_embedding_layer[0])
    t_module = nn.Sequential(
        nn.Linear(model.time_dim, model.t_embedding_layer[0])  # ,
        # nn.ReLU(),
        # nn.Dropout(p=0.5),
        # nn.Linear(model.t_embedding_layer[0], model.t_embedding_layer[0])
    )
    return x_module, t_module


def create_output_marker_layer(model):
    embed_module = nn.Sequential(
        nn.Linear(model.hidden_embed_input_dim,
                  model.shared_output_layers[0])
        # nn.Linear(
        #    self.shared_output_layers[0], self.shared_output_layers[1]), nn.ReLU()
    )

    x_module_logvar = None
    l = model.shared_output_layers[-1]
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


def create_output_time_layer(model, b, ti):
    if model.time_loss == 'intensity':
        h_influence = nn.Linear(model.shared_output_layers[-1], 1, bias=False)
        time_influence = nn.Parameter(ti * torch.ones(1, 1, 1))  # 0.005*
        base_intensity = nn.Parameter(torch.zeros(1, 1, 1) - b)  # -8
        model.h_influence, model.time_influence, model.base_intensity = h_influence, time_influence, base_intensity
    else:
        model.time_mu = nn.Linear(model.shared_output_layers[-1], 1)
        model.time_logvar = nn.Linear(model.shared_output_layers[-1], 1)
    return


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
            h : Tensor of shape (T+1)xBSxself.shared_output_layers[-1]
            t : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
                [i,:,1] represents time gap d_i = t_i- t_{i-1}
        Output:
            log_f_t : tensor of shape TxBS

    """
    h_trimmed = h  # TxBSxself.shared_output_layers[-1]
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
            phi_x : Tensor of shape TxBSx self.x_embedding_layer[-1]
            phi_t : Tensor of shape TxBSx self.t_embedding_layer[-1]
            phi   : Tensor of shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
    """
    phi_x = model.embed_x(x)
    phi_t = t
    phi = torch.cat([phi_x, phi_t], -1)
    return phi_x, phi_t, phi


def generate_marker(model, h, t):
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
    if model.x_given_t:
        d_js = t[:, :, 1][:, :, None]  # Shape TxBSx1 Time differences
        h_trimmed = torch.cat([h_trimmed, d_js], -1)

    marker_out_mu = model.output_x_mu(h_trimmed)

    if model.marker_type == 'real':
        marker_out_logvar = model.output_x_logvar(h_trimmed)
    else:
        marker_out_logvar = None
    return marker_out_mu, marker_out_logvar
