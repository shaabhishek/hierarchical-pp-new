from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.distributions import Normal, Categorical, Gumbel

from hyperparameters import EncoderHyperParams, Model1HyperParams
from utils.helper import assert_shape, pretty_print_table, prepend_dims_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel_softmax(logits, temperature, n_sample):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar

        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = n_sample x BS x k
    """
    g = Gumbel(torch.zeros(*logits.shape), torch.ones(*logits.shape)).rsample((n_sample,)).to(device)
    h = (g + logits) / temperature
    y = F.softmax(h, dim=-1)
    return y


DistSampleTuple = namedtuple('DistSampleTuple', ['dist', 'sample'])


class MLP(nn.Module):
    def __init__(self, dims: list):
        assert len(dims) >= 2  # should at least be [input_dim, output_dim]
        super().__init__()
        layers = list()
        for i in range(len(dims) - 1):
            n = dims[i]
            m = dims[i + 1]
            layer = nn.Linear(n, m, bias=True)
            layers.append(layer)
            layers.append(nn.LayerNorm(m))
            layers.append(nn.ReLU())  # NOTE: Always slaps a non-linearity in the end
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPNormal(MLP):
    def __init__(self, dims: list):
        try:
            assert len(dims) >= 3  # should at least be [inputdim, hiddendim1, outdim]
        except AssertionError:
            print(dims)
            raise

        super().__init__(dims[:-1])  # initializes the core network
        self.mu_module = nn.Linear(dims[-2], dims[-1], bias=False)
        self.logvar_module = nn.Linear(dims[-2], dims[-1], bias=False)

    def forward(self, x):
        h = self.net(x)
        mu, logvar = self.mu_module(h), self.logvar_module(h)
        dist = Normal(mu, logvar.div(2).exp())  # std = exp(logvar/2)
        return dist


class MLPCategorical(MLP):
    def __init__(self, dims: list):
        try:
            # should at least be [inputdim, hiddendim1, logitsdim] - otherwise it's just a matrix multiplication
            assert len(dims) >= 3, f"MLP dimension should at least be [in_dim, hiddendim1, out_dim], provided: {dims}"
        except AssertionError:
            print(dims)
            raise

        super().__init__(dims[:-1])  # initializes the core network
        self.logit_module = nn.Linear(dims[-2], dims[-1], bias=False)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        logits = self.logit_module(h)
        dist = Categorical(logits=logits)
        return dist


class BaseEncoder(nn.Module):
    def __init__(self, encoder_hyperparams: EncoderHyperParams):
        super().__init__()
        self.num_posterior_samples = encoder_hyperparams.num_posterior_samples
        self.rnn_hidden_dim = encoder_hyperparams.rnn_dims[-1]
        self.y_dim = encoder_hyperparams.y_dims[-1]
        self.latent_dim = encoder_hyperparams.z_dims[-1]
        self.y_module, self.rnn_module, self.z_module = self.create_inference_nets(encoder_hyperparams.y_dims,
                                                                                   encoder_hyperparams.rnn_dims,
                                                                                   encoder_hyperparams.z_dims)

    @staticmethod
    def create_inference_nets(y_dims, rnn_dims, z_dims):
        y_module = MLPCategorical(y_dims)
        rnn = BaseModel.create_rnn(rnn_dims)
        z_module = MLPNormal(z_dims)
        return y_module, rnn, z_module

    def forward(self, xt, temp, mask):
        raise NotImplementedError

    def _get_encoded_y(self, augmented_hidden_seq: torch.Tensor, mask: torch.Tensor, T: int, BS: int, temp: float,
                       Ny: int):
        """
        Encoder for y - discrete latent state

        :param Ny:
        :param augmented_hidden_seq:
        :param mask:
        :param T:
        :param BS:
        :param temp:
        :return: dist_y, sample_y
        sample_y: N x T x BS x y_dim
        """
        hidden_seq = augmented_hidden_seq[1:]
        # Need the last time index of each sequence based on mask
        last_time_idxs = torch.argmax(mask, dim=0)  # (BS,)
        # Pick out the state corresponding to last time step for each batch data point
        last_time_hidden_state = torch.cat(
            [hidden_seq[last_time_idxs[i], i][None, :] for i in range(BS)],
            dim=0)  # (BS,rnn_hidden_dim)
        assert_shape("last timestep's hidden state", last_time_hidden_state.shape, (BS, self.rnn_hidden_dim))

        dist_y = self.y_module(last_time_hidden_state)  # shape(dist_y.logits) = (BS, y_dim)
        sample_y = sample_gumbel_softmax(dist_y.logits, temp, Ny)  # (N, BS, y_dim)
        sample_y = sample_y.unsqueeze(1).expand(Ny, T, -1, -1)  # (N, T, BS,y_dim)
        assert_shape("sample_y", sample_y.shape, (Ny, T, BS, self.y_dim))
        return dist_y, sample_y


class BaseDecoder(nn.Module):
    def __init__(self, decoder_hyperparams):
        super().__init__()
        self.marker_type = decoder_hyperparams.marker_type

    def forward(self, augmented_hidden_seq, posterior_sample_z, posterior_sample_y, t, x):
        """
        :param augmented_hidden_seq: T x BS x h_dim
        :param posterior_sample_z: Nz x Ny x T x BS x latent_dim
        :param posterior_sample_y: Ny x T x BS x y_dim
        :param t: T x BS x t_dim=2
        time data
        :param x: T x BS
        marker data

        :return: marker_log_likelihood: T x BS
        :return: time_log_likelihood: T x BS
        :return: predicted_times: Nz x Ny x T x BS x 1
        :return: dist_marker_recon_logits: Nz x Ny x T x BS x x_dim
        """
        T, BS, _ = t.shape
        time_intervals, event_times = _get_timestamps_and_intervals_from_data(t)

        phi_hzy_seq = self.preprocess_latent_states(augmented_hidden_seq, posterior_sample_y, posterior_sample_z, T, BS)
        dist_marker_recon = self._get_marker_distribution(phi_hzy_seq)  # logits: (Nz, Ny, T, BS, x_dim)
        marker_log_likelihood, time_log_likelihood = self._compute_log_likelihoods(phi_hzy_seq, time_intervals,
                                                                                   x, dist_marker_recon, T, BS)

        predicted_times = self._get_predicted_times(phi_hzy_seq, event_times, BS)

        return marker_log_likelihood, time_log_likelihood, predicted_times, dist_marker_recon.logits

    def _get_marker_distribution(self, h):
        if self.marker_type == 'categorical':
            logits = self.marked_point_process_net.get_next_event_marker_logits(h)
            return Categorical(logits=logits)
        else:
            raise NotImplementedError

    def _compute_log_likelihoods(self, *args):
        raise NotImplementedError

    def _get_predicted_times(self, phi_hzy_seq, event_times, BS):
        """
        Predicts the next event for each time step using Numerical Integration

        :param phi_hzy_seq: Nz x Ny x T x BS x phi_dim
        [phi_0, .., phi_{T-1}] = f[(h_prime, z_0, y), (h_0, z_1, y), ..., (h_{T-2}, z_{T-1}, y)]
        :param event_times: T x BS x 1
        [t_0, ..., t_{T-1}]
        :param BS: int, Batch size
        :return: predicted_times = Nz x Ny x T x BS x 1
        [t'_0, ..., t'_{T-1}]
        """
        Nz, Ny = phi_hzy_seq.shape[:2]
        expanded_event_times = prepend_dims_to_tensor(event_times[:-1], Nz, Ny)  # Nz, Ny, T-1, BS, 1
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # The pairs of (h,t) should be (h_j, t_j) where h_j has information about t_j
            # don't need the predicted timestamp after the last observed event at time T
            next_event_times = self.marked_point_process_net.get_next_event_times(phi_hzy_seq[:, :, 1:],
                                                                                  expanded_event_times)  # (*, T-1, BS, 1)
            predicted_times = torch.cat([torch.zeros(Nz, Ny, 1, BS, 1).to(device), next_event_times], dim=2)
        return predicted_times


class BaseModel(nn.Module):
    model_name = ""
    encoder_hyperparams_class = None
    decoder_hyperparams_class = None

    def __init__(self, model_hyperparams):
        super().__init__()
        self.marker_type = model_hyperparams.marker_type
        self.marker_dim = model_hyperparams.marker_dim
        self.time_dim = model_hyperparams.time_dim
        self.rnn_hidden_dim = model_hyperparams.rnn_hidden_dim
        self.latent_dim = model_hyperparams.latent_dim
        self.cluster_dim = model_hyperparams.cluster_dim

        # Embedding network
        self.embed_x, self.embed_t = self.create_embedding_nets(model_hyperparams.marker_embedding_net_dims,
                                                                model_hyperparams.time_embedding_net_dims)

    def print_model_info(self):
        # dump whatever str(nn.Module) has to offer
        print(self)

        # dump parameter info
        pretty_print_table("Parameter Name", "Size")
        for param_name, param_data in self.named_parameters():
            pretty_print_table(param_name, param_data.size())

    def create_embedding_nets(self, marker_embedding_net_dims, time_embedding_net_dims):
        # marker_dim is passed. time series dim is 2
        if self.marker_type == 'categorical':
            x_module = nn.Embedding(*marker_embedding_net_dims)
        else:
            x_module = MLP(marker_embedding_net_dims)

        t_module = MLP(time_embedding_net_dims)
        return x_module, t_module

    def compute_metrics(self, predicted_times: torch.Tensor, event_times: torch.Tensor, marker_logits: torch.Tensor,
                        marker: torch.Tensor, mask: torch.Tensor):
        """
        Input:
            :param predicted_times: Nz x Ny x T x BS x 1  or T x BS x 1 ;
            pt_j : predicted time of event j
            :param event_times : T x BS x 1
            t_j : actual time of event j
            :param marker_logits : * x T x BS x marker_dim;
            :param marker : T x BS
            :param mask: T x BS
        Output:
            metric_dict: dict
        """
        metric_dict = {}
        has_latent_samples = predicted_times.dim() == 5
        if not has_latent_samples:
            predicted_times = prepend_dims_to_tensor(predicted_times, 1, 1)
            marker_logits = prepend_dims_to_tensor(marker_logits, 1, 1)
        Nz, Ny = predicted_times.shape[:2]
        num_total_events = Nz * Ny * mask[1:, :].sum().detach().cpu().numpy()
        with torch.no_grad():
            # note: both t[0] and pt[0] are 0 by convention
            time_squared_error = torch.pow((predicted_times - event_times) * mask.unsqueeze(-1),
                                           2.)  # (Nz, Ny, T, BS, 1)
            metric_dict['time_mse'] = time_squared_error.sum().cpu().numpy()
            # note: because 0th timestamps are zero always, reducing count to ensure accuracy stays unbiased
            metric_dict['time_mse_count'] = num_total_events

            if self.marker_type == "categorical":
                predicted = torch.argmax(marker_logits, dim=-1)  # (Nz, Ny, T, BS)
                correct_predictions = (predicted == marker) * mask  # (Nz, Ny, T, BS)
                correct_predictions = correct_predictions[:, :, 1:]  # Keep only the predictions from 2nd timestamp

                # count how many correct predictions we made
                metric_dict['marker_acc'] = correct_predictions.sum().cpu().numpy()
                # count how many predictions we made
                metric_dict['marker_acc_count'] = num_total_events
            else:
                raise NotImplementedError

        return metric_dict

    @staticmethod
    def augment_hidden_sequence(h_prime, hidden_seq):
        """
            Append h_prime to h_0 .. h_{T-1}
            NOTE: h_j = f(t_{j}, h_{j-1}) => the first 'h' that has saw t_j
            this is important to note for computing intensity / next event's time
            """
        hidden_seq = torch.cat([h_prime, hidden_seq], dim=0)  # (T+1, BS, rnn_hidden_dim)
        return hidden_seq

    @classmethod
    def from_model_hyperparams(cls, model_hyperparams: Model1HyperParams):
        encoder_hyperparams = cls.encoder_hyperparams_class(model_hyperparams)
        decoder_hyperparams = cls.decoder_hyperparams_class(model_hyperparams)
        return cls(model_hyperparams, encoder_hyperparams, decoder_hyperparams)

    @staticmethod
    def create_rnn(rnn_dims):
        rnn = nn.GRU(input_size=rnn_dims[0], hidden_size=rnn_dims[1])
        return rnn

    def preprocess_inputs(self, x, t):
        # Transform markers and timesteps into the embedding spaces
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1)  # (T, BS, emb_dim)
        return phi_xt


def one_hot_encoding(y, n_dims=None):
    # Implement
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def assert_input(self):
    assert self.marker_type in {'categorical'}, "Unknown Input type provided!"


def create_mlp(dims: list):
    layers = list()
    for i in range(len(dims) - 1):
        n = dims[i]
        m = dims[i + 1]
        L = nn.Linear(n, m, bias=True)
        layers.append(L)
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def _get_timestamps_and_intervals_from_data(time_data_tensor):
    time_intervals = time_data_tensor[:, :, 0:1]  # (T, BS, 1)
    event_times = time_data_tensor[:, :, 1:2]  # (T, BS, 1)
    return time_intervals, event_times
