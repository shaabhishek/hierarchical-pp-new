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
from base_model import BaseModel, MarkedPointProcessRMTPPModel
from base_model import compute_marker_log_likelihood, compute_point_log_likelihood, create_output_nets, generate_marker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False

#Move it to utils
from utils.metric import get_marker_metric, compute_time_expectation, get_time_metric

class RMTPP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = self.create_rnn()
        mpp_config = {
            "input_dim":self.shared_output_dims[-1],
            "marker_dim": self.marker_dim,
            "marker_type": self.marker_type,
            "init_base_intensity": self.base_intensity,
            "init_time_influence": self.time_influence,
            "x_given_t": self.x_given_t,
        }
        self.marked_point_process_net = MarkedPointProcessRMTPPModel(**mpp_config)
    
    def create_rnn(self):
        rnn = nn.GRU(
            input_size=self.x_embedding_dim[-1]+self.t_embedding_dim[-1],
            hidden_size=self.rnn_hidden_dim,
        )
        return rnn

    def forward(self, marker_seq, time_seq, mask=None, preds_file=None, **kwargs):
        time_log_likelihood, marker_log_likelihood, metric_dict = self._forward(marker_seq, time_seq, mask)

        marker_loss = (-1.* marker_log_likelihood * mask)[1:,:].sum()
        time_loss = (-1. *time_log_likelihood * mask)[1:,:].sum()

        loss = time_loss + marker_loss

        meta_info = {
            "marker_ll": marker_loss.detach().cpu(),
            "time_ll": time_loss.detach().cpu(),
            "true_ll": loss.detach().cpu()
        }

        return loss, {**meta_info, **metric_dict}

    def _forward(self, x, t, mask):
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1) #(T,BS, emb_dim)
        time_intervals = t[:, :, 0:1] #(T, BS, 1)
        event_times = t[:, :, 1:2] #(T, BS, 1)
        T,BS,_ = phi_x.shape

        # Run RNN over the concatenated embedded sequence
        h_0 = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        hidden_seq, _ = self.rnn(phi_xt, h_0)
        # Append h_0 to h_1 .. h_T
        # NOTE: h_j = f(t_{j-1}, h_{j-1}) => the first 'h' that has info about t_j is h_{j+1}; j = 1, ..., T
        # this is important to note for computing intensity / next event's time
        hidden_seq = torch.cat([h_0, hidden_seq], dim=0) #(T+1, BS, rnn_hidden_dim)

        # Generate marker for next_event
        # x_j = f(h_j), where j = 0, 1, ..., T; Also: h_j = f(t_{j-1})
        # NOTE: For prediction, only keep the first T markers (we don't have ground truth for {T+1}th marker)
        # Also: The first marker only depends on h0, and therefore its 'loss' will only matter if h0 is learned.
        next_marker_logits = self.marked_point_process_net.get_next_event(hidden_seq) #x_logits = (T+1, BS, marker_dim)
        predicted_marker_logits = next_marker_logits[:-1] #(T, BS, marker_dim)
        predicted_marker_dist = torch.distributions.Categorical(logits=predicted_marker_logits)

        # Compute Log Likelihoods
        time_log_likelihood = self.marked_point_process_net.get_point_log_density(hidden_seq[1:], time_intervals)
        marker_log_likelihood = self.marked_point_process_net.get_marker_log_prob(x, predicted_marker_dist)

        with torch.no_grad():
            next_event_times = self.marked_point_process_net.get_next_time(hidden_seq[1:], event_times, num_samples=5) # (T, BS, 1)
            predicted_times = torch.cat([torch.zeros(1, BS, 1).to(device), next_event_times], dim=0)[:-1] #(T+1, BS, 1)

        metric_dict = self.compute_metrics(predicted_marker_logits, predicted_times, x, event_times, mask)

        return time_log_likelihood, marker_log_likelihood, metric_dict


# class rmtpp(nn.Module):
#     """
#         Implementation of Recurrent Marked Temporal Point Processes: Embedding Event History to Vector
#         'https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf'

#         ToDo:
#             1. Mask is implemented. Verify
#             2. for categorical, x is TxBSx1. create embedding layer with one hot vector

#     """

#     def __init__(self, marker_type='real', marker_dim=31, time_dim=2, dropout = 0., hidden_dim=128, x_given_t=False,base_intensity = 0.,time_influence = 0.1, gamma = 1., time_loss = 'intensity' ):
#         super().__init__()
#         """
#             Input:
#                 marker_type : 'real' or 'binary', 'categorical'
#                 marker_dim  : number of dimension  in case of real input. Otherwise number of classes for categorical
#                 but marker_dim is 1 (just the class label)
#                 hidden_dim : hidden dimension is gru cell
#                 x_given_t : whether to condition marker given time gap. For RMTPP set it false.
#         """
#         self.model_name = 'rmtpp'#Use this for all model to decide run time behavior
#         self.marker_type = marker_type
#         self.marker_dim = marker_dim
#         self.time_dim = time_dim
#         self.hidden_dim = hidden_dim
#         self.x_given_t = x_given_t
#         self.dropout = dropout
#         self.gamma = gamma
#         self.time_loss = time_loss
#         self.use_rnn_cell = False
#         assert_input(self)

#         self.sigma_min = 1e-10

#         # Set up layer dimensions
#         self.x_embedding_layer = [256]
#         self.t_embedding_layer = [8]
#         self.shared_output_layers = [256]
#         self.hidden_embed_input_dim = self.hidden_dim 

#         # setup layers
#         self.embed_x, self.embed_time = create_input_embedding_layer(self)
#         if self.use_rnn_cell:
#             self.rnn_cell = nn.GRUCell(
#                 input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1], hidden_size=self.hidden_dim)
#         else:
#             self.rnn = nn.GRU(
#                 input_size=self.x_embedding_layer[-1] + self.t_embedding_layer[-1],
#                 hidden_size = self.hidden_dim
#                 #nonlinearity='relu'
#             )
#         # For tractivility of conditional intensity time module is a dic where forward needs to be defined
#         self.embed_hidden_state, self.output_x_mu, self.output_x_logvar = create_output_marker_layer(self)
#         create_output_time_layer(self, base_intensity, time_influence)


#     def forward(self, x, t,anneal = 1., mask= None, preds_file=None):
#         """
#             Input: 
#                 x   : Tensor of shape TxBSxmarker_dim (if real)
#                      Tensor of shape TxBSx1(if categorical)
#                 t   : Tensor of shape TxBSxtime_dim. [i,:,0] represents actual time at timestep i ,\
#                     [i,:,1] represents time gap d_i = t_i- t_{i-1}
#                 mask: Tensor of shape TxBS If mask[t,i] =1 then that timestamp is present
#             Output:
#                 loss : Tensor scalar
#                 meta_info : dict of results
#         """
#         #TxBS and TxBS
#         time_log_likelihood, marker_log_likelihood, metric_dict = self._forward(
#             x, t, mask, preds_file)
        
#         marker_loss = (-1.* marker_log_likelihood *mask)[1:,:].sum()
#         time_loss = (-1. *time_log_likelihood *mask)[1:,:].sum()


#         loss = self.gamma*time_loss + marker_loss
#         true_loss = time_loss + marker_loss
#         #if mask is not None:
#         #    loss = loss * mask
#         meta_info = {"marker_ll":marker_loss.detach().cpu(), "time_ll":time_loss.detach().cpu(), "true_ll": true_loss.detach().cpu()}
#         return loss, {**meta_info, **metric_dict}

#     def run_forward_rnn(self, x, t):
#         """
#             Input: 
#                 x   : Tensor of shape TxBSxmarker_dim (if real)
#                      Tensor of shape TxBSx1(if categorical)
#                 t   : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
#                     [i,:,1] represents time gap d_i = t_i- t_{i-1}
#             Output:
#                 Output hiddent states to be used for prediction. Thus including h_0
#                 h : Tensor of shape (T)xBSxhidden_dim
#                 embed_h : Tensor of shape (T)xBSxself.shared_output_layers[-1]
#         """
#         batch_size, seq_length = x.size(1), x.size(0)
#         # phi Tensor shape TxBS x (self.x_embedding_layer[-1] + self.t_embedding_layer[-1])
#         _, _, phi = preprocess_input(self, x, t)

#         if self.use_rnn_cell is False:
#             # Run RNN over the concatenated sequence [marker_seq_emb, time_seq_emb]
#             h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
#             hidden_seq, _ = self.rnn(phi, h_0)
#             h = torch.cat([h_0, hidden_seq], dim = 0)

#         else:
#             outs = []
#             h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
#             outs.append(h_t[None, :, :])
#             for seq in range(seq_length):
#                 h_t = self.rnn_cell(phi[seq, :, :], h_t)
#                 outs.append(h_t[None, :, :])
#             h = torch.cat(outs, dim=0)  # shape = [T+1, batchsize, h]
#         return h[:-1,:,:], self.preprocess_hidden_state(h)[:-1,:,:]

#     def preprocess_hidden_state(self, h):
#         return self.embed_hidden_state(h)

#     def compute_hidden_states(self, x, t, mask):
#         """
#         Input: 
#                 x   : Tensor of shape TxBSxmarker_dim (if real)
#                      Tensor of shape TxBSx1(if categorical)
#                 t   : Tensor of shape TxBSxtime_dim. [i,:,0] represents actual time at timestep i ,\
#                     [i,:,1] represents time gap d_i = t_i- t_{i-1}
                
#                 mask: does not matter
#         Output:
#                 hz : Tensor of shape (T)xBSxself.shared_output_layers[-1]
#         """
#         _, hidden_states = self.run_forward_rnn(x, t)
#         return hidden_states


#     def _forward(self, x, t, mask, preds_file):
#         """
#             Input: 
#                 x   : Tensor of shape TxBSxmarker_dim (if real)
#                      Tensor of shape TxBSx1(if categorical)
#                 t   : Tensor of shape TxBSxtime_dim [i,:,0] represents actual time at timestep i ,\
#                     [i,:,1] represents time gap d_i = t_i- t_{i-1}
#                 mask: Tensor of shape TxBSx1. If mask[t,i,0] =1 then that timestamp is present
#             Output:

#         """

#         # Tensor of shape (T)xBSxself.shared_output_layers[-1]
#         _, hidden_states = self.run_forward_rnn(x, t)
#         T, bs = x.size(0), x.size(1)

#         # marker generation layer. Ideally it should include time gap also.
#         # Tensor of shape TxBSx marker_dim
#         marker_out_mu, marker_out_logvar = generate_marker(self, 
#             hidden_states, t)

#         metric_dict = {}
#         time_log_likelihood, mu_time = compute_point_log_likelihood(self,
#             hidden_states, t)
#         with torch.no_grad():
#             if self.time_loss == 'intensity':
#                 mu_time = compute_time_expectation(self, hidden_states, t, mask)[:,:, None]
#             get_marker_metric(self.marker_type, marker_out_mu, x, mask, metric_dict)
#             get_time_metric(mu_time,  t, mask, metric_dict)

#             if preds_file is not None:
#                 import pdb; pdb.set_trace()
#                 if len(mu_time.data.size()) == 3:
#                     np.savetxt(preds_file, (mu_time[1:,:,0]*mask[1:, :]).cpu().numpy().T)
#                 else:
#                     np.savetxt(preds_file, (torch.mean(mu_time, dim =1)[1:,:,0]*mask[1:, :]).cpu().numpy().T)
                
#         #Pad initial Time point with 0
#         #zero_pad = torch.zeros(1, bs).to(device)
#         #time_log_likelihood = torch.cat([zero_pad, time_log_likelihood[1:,:]], dim =0)
#         marker_log_likelihood = compute_marker_log_likelihood(self, 
#             x, marker_out_mu, marker_out_logvar)

#         return time_log_likelihood, marker_log_likelihood, metric_dict  # TxBS and TxBS

        


if __name__ == "__main__":
    model = rmtpp()



