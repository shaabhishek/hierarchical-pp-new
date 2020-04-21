import torch

import torch.nn as nn
from torch.distributions.normal import Normal
from base_model import BaseModel
from marked_pp_rmtpp_model import MarkedPointProcessRMTPPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False

# Move it to utils


class RMTPP(BaseModel):
    model_name = "rmtpp"

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
            "mc_integration_num_samples": kwargs["mc_integration_num_samples"]
        }
        self.marked_point_process_net = MarkedPointProcessRMTPPModel(**mpp_config)

    def create_rnn(self):
        rnn = nn.GRU(
            input_size=self.x_embedding_dim[-1] + self.t_embedding_dim[-1],
            hidden_size=self.rnn_hidden_dim,
        )
        return rnn

    def forward(self, marker_seq, time_seq, mask=None, **kwargs):
        time_log_likelihood, marker_log_likelihood, metric_dict = self._forward(marker_seq, time_seq, mask)

        marker_nll = (-1. * marker_log_likelihood * mask)[1:, :].sum()
        time_nll = (-1. * time_log_likelihood * mask)[1:, :].sum()

        loss = time_nll + marker_nll

        meta_info = {
            "marker_nll": marker_nll.detach().cpu(),
            "time_nll": time_nll.detach().cpu(),
            "true_nll": loss.detach().cpu()
        }

        return loss, {**meta_info, **metric_dict}

    def _forward(self, x, t, mask):
        T, BS, _ = t.shape
        # Note: this hidden_seq is [h_prime, h_0, ..., h_{T-1}]
        hidden_seq, event_times, time_intervals = self.get_hidden_states_from_input(x, t)

        """
        Generate marker for next_event
        x_j = f(h_{j-1}}), where j = 0, 1, ..., T; Also: h_j = f(t_j)
        NOTE: For prediction, only keep the first T markers (we don't have ground truth for {T+1}th marker)
        NOTE: The first marker only depends on h0, and therefore its 'loss' will only matter if h0 is learned.
        """
        next_event_marker_logits = self.marked_point_process_net.get_next_event_marker_logits(
            preceding_hidden_states=hidden_seq)  # x_logits = (T+1, BS, marker_dim)
        predicted_marker_logits = next_event_marker_logits[:-1]  # (T, BS, marker_dim)
        predicted_marker_dist = torch.distributions.Categorical(logits=predicted_marker_logits)

        marker_log_likelihood, time_log_likelihood = self._compute_log_likelihoods(hidden_seq, time_intervals, x,
                                                                                   predicted_marker_dist)

        with torch.no_grad():
            # The pairs of (h,t) should be (h_j, t_j) where h_j is f(t_j)
            next_event_times = self.marked_point_process_net.get_next_event_times(hidden_seq[1:],
                                                                                  event_times)  # (T, BS, 1)
            predicted_times = torch.cat([torch.zeros(1, BS, 1).to(device), next_event_times], dim=0)[
                              :-1]  # don't need the predicted timestamp after the last observed event (T, BS, 1)

        metric_dict = self.compute_metrics(predicted_times, event_times, predicted_marker_logits, x, mask)

        return time_log_likelihood, marker_log_likelihood, metric_dict

    def _compute_log_likelihoods(self, hidden_seq, time_intervals, marker_seq, predicted_marker_dist):
        """
            :param hidden_seq: [h_prime, h0, ..., h_{T-1}]
            :param time_intervals: [i_0, ..., i_{T-1}]
            :param marker_seq: [x_0, ..., x_{T-1}]
            :param predicted_marker_dist: [fx_0, ..., fx_{T-1}]

            Compute Log Likelihoods
            Relationship between ll and (h,t,i)
            `logf*(t_{j+1}) = g(h_j, (t_{j+1}-t_j)) = g(h_j, i_{j+1})`
            => `logf*(t0) = g(h', i0)`
            and `logf*(t_{T-1}) = g(h_{T-2}, i_{T-1})`
            and finally `logf*(t_T) = g(h_{T-1}, i_T)`
            Boundary Conditions:
            logf*(t0) is the likelihood of the first event but it's not based on past
            information, so we don't use it in likelihood computation (forward function)
            logf*(t_T) is the likelihood of the next event after last observed event,
            so we don't use it either (as we don't have its corresponding timestamp)
        """
        time_log_likelihood = self.marked_point_process_net.get_point_log_density(hidden_seq[:-1], time_intervals)
        marker_log_likelihood = self.marked_point_process_net.get_marker_log_prob(marker_seq, predicted_marker_dist)
        return marker_log_likelihood, time_log_likelihood

    def get_hidden_states_from_input(self, x, t):
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1)  # (T,BS, emb_dim)
        time_intervals = t[:, :, 0:1]  # (T, BS, 1)
        event_times = t[:, :, 1:2]  # (T, BS, 1)
        T, BS, _ = phi_x.shape
        # Run RNN over the concatenated embedded sequence
        h_prime = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        hidden_seq, _ = self.rnn(phi_xt, h_prime)
        hidden_seq = self.augment_hidden_sequence(h_prime, hidden_seq)
        return hidden_seq, event_times, time_intervals


if __name__ == "__main__":
    model = RMTPP()
