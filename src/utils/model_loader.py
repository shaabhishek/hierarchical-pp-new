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

def load_model(params):
    if params.model == 'rmtpp':
<<<<<<< HEAD
        model = rmtpp(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss)
    if params.model == 'rnnbptt':
        model = rnnbptt(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss)
=======
        model = rmtpp(marker_type= params.marker_type,dropout=params.dropout, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss)
>>>>>>> f5b7a0b4c6598462c985a83d2c334a608ab9b32f
    if params.model == 'hrmtpp':
        model = hrmtpp(marker_type= params.marker_type, marker_dim = params.marker_dim, time_dim=params.time_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t, base_intensity = params.base_intensity, time_influence = params.time_influence, gamma = params.gamma, time_loss = params.time_loss)
    if params.model == 'ACD':
        model = ACD()
    return model