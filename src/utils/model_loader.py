import torch
import sys
sys.path.insert(0, './../')
from rmtpp import rmtpp
from hrmtpp import hrmtpp
from hrmtpp_exact import hrmtpp_exact
from hrmtpp_softmax import hrmtpp_softmax
from storn import storn
from h_storn_softmax import h_storn_softmax

def load_model(params):
    if params.model == 'rmtpp':
        model = rmtpp(marker_type= params.marker_type, marker_dim = params.marker_dim, hidden_dim = params.hidden_dim,x_given_t = params.x_given_t)
    return model