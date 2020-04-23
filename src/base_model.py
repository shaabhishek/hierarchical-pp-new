import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Gumbel

from utils.helper import _assert_shape

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
            assert len(
                dims) >= 3  # should at least be [inputdim, hiddendim1, logitsdim] - otherwise it's just a matrix multiplication
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
    def __init__(self, rnn_dims: list, y_dims: list, z_dims: list, num_posterior_samples:int):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.y_dims = y_dims
        self.z_dims = z_dims
        self.rnn_hidden_dim = rnn_dims[-1]
        self.y_dim = y_dims[-1]
        self.latent_dim = z_dims[-1]
        self.num_posterior_samples = num_posterior_samples
        self.y_module, self.rnn_module, self.z_module = self.create_inference_nets()

    def create_inference_nets(self):
        y_module = MLPCategorical(self.y_dims)

        rnn = nn.GRU(
            input_size=self.rnn_dims[0],
            hidden_size=self.rnn_dims[1],
        )
        z_module = MLPNormal(self.z_dims)
        return y_module, rnn, z_module

    def forward(self, xt, temp, mask):
        raise NotImplementedError

    def _get_encoded_y(self, augmented_hidden_seq: torch.Tensor, mask: torch.Tensor, T: int, BS: int, temp: float, Ny: int):
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
        _assert_shape("last timestep's hidden state", last_time_hidden_state.shape, (BS, self.rnn_hidden_dim))

        dist_y = self.y_module(last_time_hidden_state)  # shape(dist_y.logits) = (BS, y_dim)
        sample_y = sample_gumbel_softmax(dist_y.logits, temp, Ny)  # (N, BS, y_dim)
        sample_y = sample_y.unsqueeze(1).expand(Ny, T, -1, -1)  # (N, T, BS,y_dim)
        _assert_shape("sample_y", sample_y.shape, (Ny, T, BS, self.y_dim))
        return dist_y, sample_y


class BaseDecoder(nn.Module):
    def __init__(self, shared_output_dims: list, marker_dim: int, decoder_in_dim: int, **kwargs):
        super().__init__()
        self.shared_output_dims = shared_output_dims
        self.marker_dim = marker_dim
        self.time_loss = kwargs['time_loss']
        self.marker_type = kwargs['marker_type']
        self.x_given_t = kwargs['x_given_t']
        self.preprocessing_module_dims = [decoder_in_dim, *self.shared_output_dims]
        self.preprocessing_module = self.create_generative_nets()

    def create_generative_nets(self):
        module = MLP(self.preprocessing_module_dims)
        return module

    def _get_marker_distribution(self, h):
        raise NotImplementedError

    def _compute_log_likelihoods(self, *args):
        raise NotImplementedError


class BaseModel(nn.Module):
    model_name = ""

    def __init__(self, **kwargs):
        super().__init__()
        self.marker_type = kwargs['marker_type']
        self.marker_dim = kwargs['marker_dim']
        self.time_dim = kwargs['time_dim']
        self.rnn_hidden_dim = kwargs['rnn_hidden_dim']
        self.latent_dim = kwargs['latent_dim']
        self.cluster_dim = kwargs['n_cluster']
        self.x_given_t = kwargs['x_given_t']
        self.time_loss = kwargs['time_loss']
        self.base_intensity = kwargs['base_intensity']
        self.time_influence = kwargs['time_influence']

        ## Preprocessing networks
        # Embedding network
        self.x_embedding_dim = [128]
        self.t_embedding_dim = [8]
        self.emb_dim = self.x_embedding_dim[-1] + self.t_embedding_dim[-1]
        self.embed_x, self.embed_t = self.create_embedding_nets()
        self.shared_output_dims = [self.rnn_hidden_dim]

        # Inference network
        if self.latent_dim is not None:
            self.encoder_z_hidden_dims = [64, 64]
            self.encoder_y_hidden_dims = [64]
            z_input_dim = self.rnn_hidden_dim + self.emb_dim + self.cluster_dim
            self.rnn_dims = [self.emb_dim, self.rnn_hidden_dim]
            self.y_dims = [self.rnn_hidden_dim, *self.encoder_y_hidden_dims, self.cluster_dim]
            self.z_dims = [z_input_dim, *self.encoder_z_hidden_dims, self.latent_dim]
        # self.encoder = Encoder(rnn_dims=rnn_dims, y_dims=y_dims, z_dims=z_dims)

    def print_parameter_info(self):
        # dump whatever str(nn.Module) has to offer
        print(self)

        # dump parameter info
        for pname, pdata in self.named_parameters():
            print(f"{pname}: {pdata.size()}")

    def create_embedding_nets(self):
        # marker_dim is passed. timeseries_dim is 2
        if self.marker_type == 'categorical':
            x_module = nn.Embedding(self.marker_dim, self.x_embedding_dim[0])
        else:
            x_module = nn.Sequential(
                nn.Linear(self.marker_dim, self.x_embedding_dim[0]),
                nn.ReLU(),
            )
            # raise NotImplementedError

        t_module = nn.Sequential(
            nn.Linear(self.time_dim, self.t_embedding_dim[0]),
            nn.ReLU()
        )
        return x_module, t_module

    def compute_metrics(self, predicted_times, event_times, marker_logits, marker, mask):
        """
        Input:
            :param predicted_times: Nz x Ny x T x BS x 1 ;
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
        Nz, Ny = predicted_times.shape[:2]
        num_total_events = Nz * Ny * mask[1:, :].sum().detach().cpu().numpy()
        with torch.no_grad():
            # note: both t[0] and pt[0] are 0 by convention
            time_squared_error = torch.pow((predicted_times - event_times) * mask.unsqueeze(-1), 2.) # (Nz, Ny, T, BS, 1)
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
