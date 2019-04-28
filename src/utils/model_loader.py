import torch
import sys
sys.path.insert(0, './../')
from rmtpp import rmtpp
from rnnbptt import rnnbptt
from hrmtpp import hrmtpp
from autoregressive import ACD
from hrmtpp_exact import hrmtpp_exact
from hrmtpp_softmax import hrmtpp_softmax
from storn import storn
from h_storn_softmax import h_storn_softmax
from model1 import Model1
from model2 import Model2
from model11 import Model11

def load_model(params):
    if params.model == 'rmtpp':
        model = rmtpp(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    # if params.model == 'rnnbptt':
    #     model = rnnbptt(marker_type= params.marker_type,bptt= params.bptt, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss)
    if params.model == 'hrmtpp':
        model = hrmtpp(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim, n_cluster=params.n_cluster, x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    if params.model == 'ACD':
        model = ACD()
    # if params.model == 'model1':
        # model = Model1(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    import pdb; pdb.set_trace()
    if params.model == 'model11':
        model = Model11(marker_type= params.marker_type, marker_dim = params.marker_dim, latent_dim=params.latent_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim, n_cluster=params.n_cluster, x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    if params.model == 'model2':
        model = Model2(marker_type= params.marker_type, marker_dim = params.marker_dim, latent_dim=params.latent_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim, n_cluster=params.n_cluster, x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss, dropout=params.dropout)
    return model
