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
from base_model import BaseModel
from marked_pp_rmtpp_model import MarkedPointProcessRMTPPModel
from base_model import compute_marker_log_likelihood, compute_point_log_likelihood, create_output_nets, generate_marker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False

# Move it to utils
from utils.metric import get_marker_metric, compute_time_expectation, get_time_metric


class RMTPP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = self.create_rnn()
        mpp_config = {
            "input_dim": self.shared_output_dims[-1],
            "marker_dim": self.marker_dim,
            "marker_type": self.marker_type,
            "init_base_intensity": self.base_intensity,
            "init_time_influence": self.time_influence,
            "x_given_t": self.x_given_t,
        }
        self.marked_point_process_net = MarkedPointProcessRMTPPModel(**mpp_config)

    def create_rnn(self):
        rnn = nn.GRU(
            input_size=self.x_embedding_dim[-1] + self.t_embedding_dim[-1],
            hidden_size=self.rnn_hidden_dim,
        )
        return rnn

    def forward(self, marker_seq, time_seq, mask=None, preds_file=None, **kwargs):
        time_log_likelihood, marker_log_likelihood, metric_dict = self._forward(marker_seq, time_seq, mask)

        marker_loss = (-1. * marker_log_likelihood * mask)[1:, :].sum()
        time_loss = (-1. * time_log_likelihood * mask)[1:, :].sum()

        loss = time_loss + marker_loss

        meta_info = {
            "marker_ll": marker_loss.detach().cpu(),
            "time_ll": time_loss.detach().cpu(),
            "true_ll": loss.detach().cpu()
        }

        return loss, {**meta_info, **metric_dict}

    def _forward(self, x, t, mask):
        T, BS, _ = t.shape
        hidden_seq, event_times, time_intervals = self.get_hidden_states_from_input(t, x)

        # Generate marker for next_event
        # x_j = f(h_j), where j = 0, 1, ..., T; Also: h_j = f(t_{j-1})
        # NOTE: For prediction, only keep the first T markers (we don't have ground truth for {T+1}th marker)
        # Also: The first marker only depends on h0, and therefore its 'loss' will only matter if h0 is learned.
        next_marker_logits = self.marked_point_process_net.get_next_event(
            hidden_seq)  # x_logits = (T+1, BS, marker_dim)
        predicted_marker_logits = next_marker_logits[:-1]  # (T, BS, marker_dim)
        predicted_marker_dist = torch.distributions.Categorical(logits=predicted_marker_logits)

        # Compute Log Likelihoods
        # logf*(t0) = f(h0, t0-0), logf*(t1) = f(h1, t1-t0), ..., logf*(t_T) = f(hT, t_T - t_{T-1})
        # logf*(t0) = f(h0, i0=0), logf*(t1) = f(h1, i1), ..., logf*(t_{T-1}) = f(h_{T-1}, i_{T-1})
        # Boundary Conditions:
        # logf*(t0) is the likelihood of the first event but it's not based on past
        # information, so we don't use it in likelihood computation (forward function)
        # logf*(t_T) is the likelihood of the next event after last observed event,
        # so we don't use it either (we don't have its corresponding timestamp)
        time_log_likelihood = self.marked_point_process_net.get_point_log_density(hidden_seq[:-1], time_intervals)
        marker_log_likelihood = self.marked_point_process_net.get_marker_log_prob(x, predicted_marker_dist)

        with torch.no_grad():
            next_event_times = self.marked_point_process_net.get_next_time(hidden_seq[1:], event_times,
                                                                           num_samples=10)  # (T, BS, 1)
            predicted_times = torch.cat([torch.zeros(1, BS, 1).to(device), next_event_times], dim=0)[
                              :-1]  # don't need the next-of-last time (T, BS, 1)

        metric_dict = self.compute_metrics(predicted_marker_logits, predicted_times, x, event_times, mask)

        return time_log_likelihood, marker_log_likelihood, metric_dict

    def get_hidden_states_from_input(self, t, x):
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1)  # (T,BS, emb_dim)
        time_intervals = t[:, :, 0:1]  # (T, BS, 1)
        event_times = t[:, :, 1:2]  # (T, BS, 1)
        T, BS, _ = phi_x.shape
        # Run RNN over the concatenated embedded sequence
        h_0 = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        hidden_seq, _ = self.rnn(phi_xt, h_0)
        # Append h_0 to h_1 .. h_T
        # NOTE: h_j = f(t_{j-1}, h_{j-1}) => the first 'h' that has info about t_j is h_{j+1}; j = 1, ..., T
        # this is important to note for computing intensity / next event's time
        hidden_seq = torch.cat([h_0, hidden_seq], dim=0)  # (T+1, BS, rnn_hidden_dim)
        return hidden_seq, event_times, time_intervals


if __name__ == "__main__":
    model = RMTPP()
