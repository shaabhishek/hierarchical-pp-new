import torch
import sys
sys.path.insert(0, './../')
from rmtpp import RMTPP
from autoregressive import ACD
from model2 import Model2
from model2_filt import Model2Filter
from model2_new import Model2New
from model1 import Model1

def load_model(params):
    if params.model == 'rmtpp':
        model = RMTPP(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, rnn_hidden_dim = params.rnn_hidden_dim, x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout, latent_dim=None, n_cluster=None, )
    if params.model == 'ACD':
        model = ACD()
    if params.model == 'model1':
        model = Model1(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, rnn_hidden_dim = params.rnn_hidden_dim, n_cluster=params.n_cluster, latent_dim=params.latent_dim, x_given_t=params.x_given_t, base_intensity=params.base_intensity, time_influence=params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    if params.model == 'model2':
        model = Model2(marker_type= params.marker_type, marker_dim = params.marker_dim, latent_dim=params.latent_dim, time_dim=params.time_dim, rnn_hidden_dim = params.rnn_hidden_dim, n_cluster=params.n_cluster, x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    if params.model == 'model2_filt':
        model = Model2Filter(n_sample = params.n_sample, marker_type= params.marker_type, marker_dim = params.marker_dim, latent_dim=params.latent_dim, time_dim=params.time_dim, rnn_hidden_dim = params.rnn_hidden_dim, n_cluster=params.n_cluster, x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    if params.model == 'model2_new':
        model = Model2New(n_sample = params.n_sample, marker_type= params.marker_type, marker_dim = params.marker_dim, latent_dim=params.latent_dim, time_dim=params.time_dim, rnn_hidden_dim = params.rnn_hidden_dim, n_cluster=params.n_cluster, x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    return model
