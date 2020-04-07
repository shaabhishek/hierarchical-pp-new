from typing import Any, Union

import torch
from torch import nn, Tensor
from torch import Tensor
from torch.nn import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_closest_hidden_states_and_timestamps(grid_times, hidden_state_sequence, timestamps):
    closest_timestamp_indices = _get_closest_timestamp_index_to_t(grid_times, timestamps).transpose(0,
                                                                                                    1)  # (BS, 1000)
    closest_timestamp = torch.stack(
        [timestamps[idxs_seq] for idxs_seq in closest_timestamp_indices], dim=1
    )  # (1000, BS)
    closest_hidden_states = torch.stack(
        [hidden_state_sequence[idxs_seq] for idxs_seq in closest_timestamp_indices], dim=1
    )  # (1000, BS, h_dim)
    return closest_hidden_states, closest_timestamp


class MarkedPointProcessRMTPPModel(nn.Module):
    def __init__(self, input_dim: int, marker_dim: int, marker_type: str, init_base_intensity: float,
                 init_time_influence: float, x_given_t: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.marker_dim = marker_dim
        self.x_given_t = x_given_t
        self.marker_type = marker_type

        if self.x_given_t:
            self.marker_input_dim = self.input_dim + 1
        else:
            self.marker_input_dim = self.input_dim

        self.create_point_net(init_base_intensity, init_time_influence)
        self.create_marker_net()

    def create_point_net(self, init_base_intensity: float, init_time_influence: float):
        self.h_influence = nn.Linear(self.input_dim, 1, bias=False)  # vector of size `input_dim`
        self.time_influence = nn.Parameter(init_time_influence * torch.ones(1))  # scalar
        self.base_intensity = nn.Parameter(torch.zeros(1) - init_base_intensity)  # scalar

    def create_marker_net(self):
        if self.marker_type == 'categorical':
            self.marker_net = nn.Linear(self.marker_input_dim, self.marker_dim)
        else:
            raise NotImplementedError

    def get_marker_log_prob(self, marker, pred_marker_dist):
        """
        Input:
            marker : Tensor of shape T x BS (if categorical); ground truth
            pred_marker_dist : Distribution object containing information about the predicted logits
        Output:
            log_prob : tensor of shape T x BS - this is the same as
        """
        T, BS = marker.shape[:2]
        if self.marker_type == "categorical":
            return pred_marker_dist.log_prob(marker)  # (T,BS)
        else:
            raise NotImplementedError

    def get_log_intensity(self, h, time_interval):
        """
        Input:
            same as for get_point_log_density function
        Output:
            log_intensity : tensor of shape TxBS - computed as per eqn 11 in the paper
        """
        past_influence, current_influence = self._get_past_and_current_influences(h, time_interval)
        log_intensity = past_influence + current_influence + self.base_intensity
        return log_intensity

    def get_point_log_density(self, h, time_intervals):
        """
        Input:
            h : Tensor of shape * x T x BS x self.shared_output_dims[-1] where '*' could possibly have more dimensions
            h_j is the first hidden state that has information about t_j
            time_intervals : Tensor of shape * x T x BS x 1  where '*' could possibly have more dimensions
        Output:
            log_prob : tensor of shape * x T x BS - computed as per eqn 12 in the paper
        """

        past_influence, current_influence = self._get_past_and_current_influences(h, time_intervals)
        log_intensity = past_influence + current_influence + self.base_intensity  # (*, T, BS, 1)
        term_2 = (past_influence + self.base_intensity).exp()  # (*, T, BS, 1)

        log_density = log_intensity + (term_2 - log_intensity.exp()) / self.time_influence

        return log_density.squeeze(-1)

    def _get_past_and_current_influences(self, h, time_interval):
        past_influence = self.h_influence(h)  # (*, T, BS, 1)
        current_influence = self.time_influence * time_interval  # (*, T, BS, 1)
        return past_influence, current_influence

    def get_intensity_over_grid(self, hidden_state_sequence, timestamps: Tensor):
        """

        :param hidden_state_sequence: Tensor of shape T x BS x h_dim
        :param timestamps:
        :return:
        """
        sequence_durations, _ = timestamps.max(dim=0).squeeze()  # (BS,)
        grid_times = torch.stack([torch.linspace(0, t, 1000) for t in sequence_durations], dim=1).to(device)  # (1000, BS)
        closest_hidden_states, closest_timestamps = extract_closest_hidden_states_and_timestamps(grid_times,
                                                                                                     hidden_state_sequence,
                                                                                                     timestamps)
        grid_times_since_last_event = (grid_times - closest_timestamps).unsqueeze(-1)  # (1000, BS, 1)
        log_intensities: Tensor = self.get_log_intensity(closest_hidden_states, grid_times_since_last_event)
        return log_intensities, grid_times

    def _mc_transformation(self, y, h, tj):
        """
        Input:
            y : N x T x BS x 1
            h : T x BS x h_dim
            tj : T x BS x 1
        Output:
            h(y): N x T x BS x 1
        """
        tj = tj.unsqueeze(0)  # (1, T, BS, 1)
        query_times = tj - 1 + 1 / y  # (N, T, BS, 1)
        time_intervals = query_times - tj  # (N, T, BS, 1)
        log_density = self.get_point_log_density(h, time_intervals).unsqueeze(-1)  # (N, T, BS, 1)
        h = (query_times * log_density.exp()) / y ** 2  # (N, T, BS, 1)
        return h

    def get_next_time(self, h, event_times, num_samples=5):
        """
        Input:
            h : Tensor of shape T x BS x self.shared_output_dims[-1]
            h_j is the first hidden state that has information about t_j
            event_times : Tensor of shape T x BS x 1
        Output:
            mu_times_next: Tensor of shape T x BS x 1
        """
        y = torch.rand(num_samples, *event_times.shape).to(
            device)  # (N, T, BS, 1) where N = number of samples for monte carlo approx
        expected_t_next = self._mc_transformation(y, h, event_times).mean(0)  # (T, BS, 1)

        return expected_t_next

    def get_next_event(self, h):
        """
        Input:
            h : Tensor of shape (T+1) x BS x self.shared_output_dims[-1]
            x_j = f(h_j), where j = 0, 1, ..., T; Also: h_j = f(t_{j-1})
        Output:
            x_logits: Tensor of shape (T+1) x BS x 1
        """
        x_logits = self.marker_net(h)
        return x_logits

    def forward(self):
        raise NotImplementedError


def _get_closest_timestamp_index_to_t(t, timestamps):
    """
    :param t:
    Tensor (N, BS) in case of separate t for each batch sequence
    :param timestamps: Tensor T x BS x 1
    :return: long Tensor of the shape (N, BS)
    """
    time_dim = 0
    T, BS, _ = timestamps.shape
    is_earlier_time = (timestamps.view(T, 1, BS) < t.view(1, *t.shape)).long()  # (T, N, BS)
    # argmax returns the last index of the largest element,
    # which is the latest 'earlier_time' in our case
    closest_times = is_earlier_time.argmax(dim=time_dim)  # (N, BS)
    return closest_times
