import torch
from torch.nn import Module
from torch.optim import Optimizer
import sys
import os
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, './../')
from rmtpp import RMTPP
from autoregressive import ACD
from model2 import Model2
from model2_filt import Model2Filter
from model2_new import Model2New
from model1 import Model1

class ModelLoader:
    def __init__(self, params, model_state_path:str=None):
        self.params = params
        self.model = self._load_model(self.params)
        if model_state_path is not None:
            self.model = self._load_model_state(self.model, model_state_path)
            self.model_state_path = Path(model_state_path)
        else:
            self.model_state_path = None

    def _load_model(self, params):
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
    
    def _load_model_state(self, model:Module, model_state_path:Path):
        checkpoint = torch.load(model_state_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

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


# def save_model(model:torch.nn.Module, optimizer:Optimizer, params:Namespace, loss):
#     state = {
#         'epoch':idx,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }

#     path = os.path.join('model', params.save, params.model, file_name)+'_'+ str(idx+1)
#     torch.save(state, path)