import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal, Categorical

from base_model import BaseEncoder, BaseDecoder, BaseModel, _get_timestamps_and_intervals_from_data
from marked_pp_rmtpp_model import MarkedPointProcessRMTPPModel
from utils.helper import _prepend_dims_to_tensor, _assert_shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model1(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma_min = 1e-10
        self.gamma = kwargs['gamma']
        self.dropout = kwargs['dropout']

        # Prior distributions
        self.prior_dist_y = Categorical(logits=torch.ones(1, self.cluster_dim).to(device))
        self.prior_dist_z = Normal(torch.tensor([0.]).to(device), torch.tensor([1.]).to(device))

        # Encoder / Inference network
        self.encoder = Model1Encoder(rnn_dims=self.rnn_dims,
                                     y_dims=self.y_dims, z_dims=self.z_dims,
                                     num_posterior_samples=kwargs['n_samples_posterior'])

        # Decoder / Generative network
        encoder_out_dim = self.rnn_hidden_dim + self.latent_dim + self.cluster_dim

        mpp_config = {
            "input_dim": self.shared_output_dims[-1],
            "marker_dim": self.marker_dim,
            "marker_type": self.marker_type,
            "init_base_intensity": self.base_intensity,
            "init_time_influence": self.time_influence,
            "x_given_t": self.x_given_t,
            "mc_integration_num_samples": kwargs["mc_integration_num_samples"]
        }
        decoder_kwargs = {'marker_dim': self.marker_dim, 'decoder_in_dim': encoder_out_dim, 'time_loss': self.time_loss,
                          'marker_type': self.marker_type, 'x_given_t': self.x_given_t, 'mpp_config': mpp_config}
        self.decoder = Model1Decoder(self.shared_output_dims, **decoder_kwargs)

    def create_output_nets(self):
        l = self.shared_output_dims[-1]
        t_module_mu = nn.Linear(l, 1)
        t_module_logvar = nn.Linear(l, 1)

        x_module_mu = None
        x_module_logvar = None

        if self.x_given_t:
            l += 1
        if self.marker_type == 'real':
            x_module_mu = nn.Linear(l, self.marker_dim)
            x_module_logvar = nn.Linear(l, self.marker_dim)
        elif self.marker_type == 'binary':  # Fix binary
            x_module_mu = nn.Sequential(
                nn.Linear(l, self.marker_dim),
                nn.Sigmoid())
        elif self.marker_type == 'categorical':
            x_module_mu = nn.Sequential(
                nn.Linear(l, self.marker_dim)  # ,
                # nn.Softmax(dim=-1)
            )
        return t_module_mu, t_module_logvar, x_module_mu, x_module_logvar

    def forward(self, marker_seq, time_seq, anneal=1., mask=None, temp=0.5):

        time_log_likelihood, marker_log_likelihood, kl, metric_dict = self._forward(marker_seq, time_seq, temp, mask)

        marker_nll = (-1. * marker_log_likelihood * mask)[1:, :].sum()
        time_nll = (-1. * time_log_likelihood * mask)[1:, :].sum()

        loss = self.gamma * time_nll + marker_nll + anneal * kl
        true_loss = time_nll + marker_nll + kl
        meta_info = {"marker_nll": marker_nll.detach().cpu(), "time_nll": time_nll.detach().cpu(),
                     "true_nll": true_loss.detach().cpu(), "kl": kl.detach().cpu()}
        return loss, {**meta_info, **metric_dict}

    def compute_metrics(self, predicted_times, event_times, marker_logits, marker, mask):
        # TODO: Implement This
        metric_dict = super(Model1, self).compute_metrics(predicted_times, event_times, marker_logits, marker, mask)
        # metric_dict["z_cluster"] = 0
        return metric_dict

    def _forward(self, x, t, temp, mask):
        time_intervals, event_times = _get_timestamps_and_intervals_from_data(t)  # (T, BS, 1)
        T, BS, _ = t.shape
        phi_x, phi_xt = self.preprocess_inputs(x, t)

        """Encoder
        Get the sampled value and (mean + var) latent variable using the hidden state sequence.
        The (augmented) hidden state sequence includes the initial hidden state (h_prime) made of zeros as well 
        """
        augmented_hidden_seq, (posterior_sample_y, posterior_dist_y), (
            posterior_sample_z, posterior_dist_z) = self.encoder(phi_xt, temp, mask)

        """Decoder
        Use the latent states computed by the encoder to compute the Log likelihood, and the reconstructed samples
        """

        marker_log_likelihood, time_log_likelihood, predicted_times, predicted_marker_logits = self.decoder(
            augmented_hidden_seq, posterior_sample_z, posterior_sample_y, t, x)  # (T,BS,shared_output_dims[-1])

        """Compute KL Divergence
        The Prior Distributions (of latent variables) are pre-computed.
        The Posterior Distributions are returned by the encoder.
        """
        # Prior distributions
        prior_dist_y, prior_dist_z = self._get_prior_distributions()
        kl = self._compute_kl(prior_dist_y, posterior_dist_y, prior_dist_z, posterior_dist_z, mask)

        metric_dict = self.compute_metrics(predicted_times, event_times, predicted_marker_logits, x, mask)
        return time_log_likelihood, marker_log_likelihood, kl, metric_dict

    def preprocess_inputs(self, x, t):
        # Transform markers and timesteps into the embedding spaces
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1)  # (T,BS, emb_dim)
        return phi_x, phi_xt

    def _get_prior_distributions(self):
        return self.prior_dist_y, self.prior_dist_z

    @staticmethod
    def _compute_kl(prior_dist_y, posterior_dist_y, prior_dist_z, posterior_dist_z, mask):
        """

        :param prior_dist_y: Distribution with logits 1 x y_dim
        :param posterior_dist_y: Distribution with logits BS x y_dim
        :param prior_dist_z: Distribution with mean/var (scalar) 0./1.
        :param posterior_dist_z: Distribution with mean/var Ny x T x BS x z_dim
        :param mask: T x BS
        :return:
        """
        kl_cluster = kl_divergence(posterior_dist_y, prior_dist_y)  # (1,BS)
        kl_latent_state = kl_divergence(posterior_dist_z, prior_dist_z).mean(0).sum(-1) * mask # (T, BS)
        kl = kl_cluster.sum() + kl_latent_state.sum()
        assert (kl >= 0), "kl should be non-negative"
        return kl


class Model1Encoder(BaseEncoder):

    def __init__(self, rnn_dims: list, y_dims: list, z_dims: list, num_posterior_samples: int):
        super().__init__(rnn_dims, y_dims, z_dims, num_posterior_samples)

    def forward(self, xt, temp, mask):
        """
        Input:
            xt: Tensor of shape T x BS x (self.x_embedding_dim[-1]+self.t_embedding_dim[-1])
            temp: scalar
            mask : Tensor TxBS
        Output:
            dist_y: Distribution with logits of shape BS x cluster_dim
            sample_y: Tensor of shape Ny x T x BS x cluster_dim
            dist_z: Distribution with mean/std of shape Ny x T x BS x latent_dim
            sample_z: Tensor of shape Nz x Ny x T x BS x latent_dim
        """
        T, BS, _ = xt.shape

        augmented_hidden_seq = self._get_encoded_h(xt, BS)
        dist_y, sample_y = self._get_encoded_y(augmented_hidden_seq, mask, T, BS, temp, self.num_posterior_samples)
        dist_z, sample_z = self._get_encoded_z(augmented_hidden_seq, xt, sample_y, self.num_posterior_samples)
        return augmented_hidden_seq, (sample_y, dist_y), (sample_z, dist_z)

    def _get_encoded_h(self, xt, BS):
        # Compute encoder RNN hidden states
        h_prime = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        hidden_seq, _ = self.rnn_module(xt, h_prime)  # (T, BS, rnn_hidden_dim)
        augmented_hidden_seq = BaseModel.augment_hidden_sequence(h_prime, hidden_seq)
        return augmented_hidden_seq

    def _get_encoded_z(self, augmented_hidden_seq, data, sample_y, Nz):
        """Encoder for z - continuous latent state
        Computes z_t <- f(y, h_{t-1}, data_t)

        :param Nz: int
        :param augmented_hidden_seq: T x BS x h_dim
        [h_prime, h_0, ..., h_{T-1}]
        :param data: T x BS x embed_dim
        [data_0, ..., data_{T-1}]
        :param sample_y: Ny x T x BS x y_dim
        This is embedded data sequence which has information for both marker and time.
        :return sample_z : Nz x Ny x T x BS x latent_dim
        Sample [z_0, ..., z_{T-1}] for each batch
        """
        Ny = self.num_posterior_samples
        concat_hxty = torch.cat(
            [
                _prepend_dims_to_tensor(augmented_hidden_seq[:-1], Ny),
                _prepend_dims_to_tensor(data, Ny),
                sample_y
            ], dim=-1
        )  # (N, T, BS, ..)
        dist_z = self.z_module(concat_hxty)  # (N,T,BS,latent_dim)
        sample_z = dist_z.rsample((Nz,))  # (n_sample, N,T,BS,latent_dim)
        return dist_z, sample_z


class Model1Decoder(BaseDecoder):
    def __init__(self, shared_output_dims: list, marker_dim: int, decoder_in_dim: int, mpp_config: dict, **kwargs):
        super().__init__(shared_output_dims, marker_dim, decoder_in_dim, **kwargs)
        self.marked_point_process_net = MarkedPointProcessRMTPPModel(**mpp_config)

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

        dist_marker_recon = self._get_marker_distribution(phi_hzy_seq)
        marker_log_likelihood, time_log_likelihood = self._compute_log_likelihoods(phi_hzy_seq, time_intervals,
                                                                                   x, dist_marker_recon, T, BS)
        predicted_times = self._get_predicted_times(phi_hzy_seq, event_times, BS)

        return marker_log_likelihood, time_log_likelihood, predicted_times, dist_marker_recon.logits

    def preprocess_latent_states(self, augmented_hidden_seq, posterior_sample_y, posterior_sample_z, T, BS):
        """
        Transforms [(h_prime, z_0, y), (h_0, z_1, y), ..., (h_{T-2}, z_{T-1}, y)] to ---> [phi_0, .., phi_{T-1}]

        :param augmented_hidden_seq: T x BS x h_dim
        [h_prime, h0, ..., h_{T-1}]
        :param posterior_sample_y: Ny x T x BS x y_dim
        [y, ..., y]
        :param posterior_sample_z: Nz x Ny x T x BS x latent_dim
        [z_0, ..., z_{T-1}]
        :return: phi_hzy_seq: Nz x Ny x T x BS x shared_output_dims[-1]
        [phi_0, ..., phi_{T-1}]
        """

        def _get_final_dim(*args):
            tensors = tuple(args)
            return tuple(tensor.shape[-1] for tensor in tensors)

        # h_dim, y_dim = _get_final_dim(augmented_hidden_seq, posterior_sample_y)
        Nz, Ny = posterior_sample_z.shape[0], posterior_sample_y.shape[0]
        concat_hzy = torch.cat(
            [
                _prepend_dims_to_tensor(augmented_hidden_seq[:-1], Nz, Ny),
                posterior_sample_z,
                _prepend_dims_to_tensor(posterior_sample_y, Nz)
            ],
            dim=-1
        )  # (Nz, Ny, T, BS, (h+z+y)_dims)
        phi_hzy_seq = self.preprocessing_module(concat_hzy)  # (Nz, Ny, T,BS,shared_output_dims[-1])
        return phi_hzy_seq

    def _get_marker_distribution(self, h):
        if self.marker_type == 'real':
            raise NotImplementedError
            # return Normal(mu, logvar.div(2).exp())
        elif self.marker_type == 'categorical':
            logits = self.marked_point_process_net.get_next_event_marker_logits(h)
            return Categorical(logits=logits)
        else:
            raise NotImplementedError

    def _compute_log_likelihoods(self, phi_hzy_seq: torch.Tensor, time_intervals: torch.Tensor,
                                 marker_seq: torch.Tensor, predicted_marker_dist: torch.distributions.Distribution,
                                 T: int, BS: int):
        """
        :param phi_hzy_seq: Nz x Ny x T x BS x shared_output_dims[-1]
        [phi_0, ..., phi_{T-1}]
        :param time_intervals: T x BS x 1
        [i_0, ..., i_{T-1}]
        :param marker_seq: T x BS
        [x_0, ..., x_{T-1}]
        :param predicted_marker_dist: distribution with logits of shape Nz x Ny x T x BS x x_dim
        [fx_0, ..., fx_{T-1}]

        :return marker_log_likelihood: T x BS
        :return time_log_likelihood: T x BS


        Computing Time Log Likelihood:
        Relationship between ll and (h,t,i)
        `logf*(t_{j+1}) = g(h_j, z_{j+1}, y, (t_{j+1}-t_j)) = g(phi_j, i_{j+1})`
        which implies
        (first timestep) `logf*(t0) = g(h', z_0, y, t_0) := g(phi_0, i_0)`
        (last timestep) `logf*(t_{T-1}) = g(phi_{T-2}, i_{T-1})`

        Boundary Conditions:
        logf*(t0) is the likelihood of the first event but it's not based on past
        information, so we don't use it in likelihood computation (forward function)

        Finally, Expectation is taken wrt posterior samples by taking the mean along first two dimensions

        """
        Nz, Ny = phi_hzy_seq.shape[:2]
        expanded_time_intervals = _prepend_dims_to_tensor(time_intervals, Nz, Ny)
        time_log_likelihood = self.marked_point_process_net.get_point_log_density(phi_hzy_seq, expanded_time_intervals)
        marker_log_likelihood = self.marked_point_process_net.get_marker_log_prob(marker_seq, predicted_marker_dist)
        time_log_likelihood_expectation = time_log_likelihood.mean((0, 1))
        marker_log_likelihood_expectation = marker_log_likelihood.mean((0, 1))
        _assert_shape("time log likelihood", time_log_likelihood_expectation.shape, (T, BS,))
        _assert_shape("marker log likelihood", marker_log_likelihood_expectation.shape, (T, BS))
        return marker_log_likelihood_expectation, time_log_likelihood_expectation

    def _get_predicted_times(self, phi_hzy_seq, event_times, BS):
        """
        Predicts the next event for each time step using MC Integration

        :param phi_hzy_seq: Nz x Ny x T x BS x phi_dim
        [phi_0, .., phi_{T-1}] = f[(h_prime, z_0, y), (h_0, z_1, y), ..., (h_{T-2}, z_{T-1}, y)]
        :param event_times: T x BS x 1
        [t_0, ..., t_{T-1}]
        :param BS: int, Batch size
        :return: predicted_times = Nz x Ny x T x BS x 1
        [t'_0, ..., t'_{T-1}]
        """
        Nz, Ny = phi_hzy_seq.shape[:2]
        expanded_event_times = _prepend_dims_to_tensor(event_times[:-1], Nz, Ny) # Nz, Ny, T-1, BS, 1
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # The pairs of (h,t) should be (h_j, t_j) where h_j has information about t_j
            # don't need the predicted timestamp after the last observed event at time T
            next_event_times = self.marked_point_process_net.get_next_event_times(phi_hzy_seq[:, :, 1:],
                                                                                  expanded_event_times) # (*, T-1, BS, 1)
            predicted_times = torch.cat([torch.zeros(Nz, Ny, 1, BS, 1).to(device), next_event_times], dim=2)
        return predicted_times
