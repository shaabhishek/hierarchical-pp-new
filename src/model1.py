import torch
from torch.distributions import kl_divergence, Normal, Categorical

from base_model import BaseEncoder, BaseDecoder, BaseModel, _get_timestamps_and_intervals_from_data, MLP
from hyperparameters import Decoder1HyperParams, Encoder1HyperParams, Model1HyperParams
from marked_pp_rmtpp_model import MarkedPointProcessRMTPPModel
from utils.helper import prepend_dims_to_tensor, assert_shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model1(BaseModel):
    model_name = "model1"
    encoder_hyperparams_class = Encoder1HyperParams
    decoder_hyperparams_class = Decoder1HyperParams

    def __init__(self, model_hyperparams: Model1HyperParams, encoder_hyperparams: Encoder1HyperParams,
                 decoder_hyperparams: Decoder1HyperParams):
        super().__init__(model_hyperparams)
        self.gamma = model_hyperparams.gamma
        self.dropout = model_hyperparams.dropout

        # Prior distributions
        self.prior_dist_y = Categorical(logits=torch.ones(1, self.cluster_dim).to(device))
        self.prior_dist_z = Normal(torch.tensor([0.]).to(device), torch.tensor([1.]).to(device))

        self.encoder = Model1Encoder(encoder_hyperparams)
        self.decoder = Model1Decoder(decoder_hyperparams)

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
        phi_xt = self.preprocess_inputs(x, t)

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
        kl_latent_state = kl_divergence(posterior_dist_z, prior_dist_z).mean(0).sum(-1) * mask  # (T, BS)
        kl = kl_cluster.sum() + kl_latent_state.sum()
        assert (kl >= 0), "kl should be non-negative"
        return kl


class Model1Encoder(BaseEncoder):
    def __init__(self, encoder_hyperparams: Encoder1HyperParams):
        super().__init__(encoder_hyperparams)

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
        return augmented_hidden_seq, (sample_y, dist_y), (sample_z, dist_z)  # TODO: move to DistSampleTuple

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
                prepend_dims_to_tensor(augmented_hidden_seq[:-1], Ny),
                prepend_dims_to_tensor(data, Ny),
                sample_y
            ], dim=-1
        )  # (N, T, BS, ..)
        dist_z = self.z_module(concat_hxty)  # (Ny,T,BS,latent_dim)
        sample_z = dist_z.rsample((Nz,))  # (Nz, Ny, T,BS,latent_dim)
        return dist_z, sample_z


class Model1Decoder(BaseDecoder):
    def __init__(self, decoder_hyperparams: Decoder1HyperParams):
        super().__init__(decoder_hyperparams)
        self.preprocessing_module = MLP(decoder_hyperparams.filtering_preprocessing_module_dims)
        self.marked_point_process_net = MarkedPointProcessRMTPPModel(**decoder_hyperparams.marked_point_process_params)

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
                prepend_dims_to_tensor(augmented_hidden_seq[:-1], Nz, Ny),
                posterior_sample_z,
                prepend_dims_to_tensor(posterior_sample_y, Nz)
            ],
            dim=-1
        )  # (Nz, Ny, T, BS, (h+z+y)_dims)
        phi_hzy_seq = self.preprocessing_module(concat_hzy)  # (Nz, Ny, T,BS,shared_output_dims[-1])
        return phi_hzy_seq

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
        expanded_time_intervals = prepend_dims_to_tensor(time_intervals, Nz, Ny)
        time_log_likelihood = self.marked_point_process_net.get_point_log_density(phi_hzy_seq, expanded_time_intervals)
        marker_log_likelihood = self.marked_point_process_net.get_marker_log_prob(marker_seq, predicted_marker_dist)
        time_log_likelihood_expectation = time_log_likelihood.mean((0, 1))
        marker_log_likelihood_expectation = marker_log_likelihood.mean((0, 1))
        assert_shape("time log likelihood", time_log_likelihood_expectation.shape, (T, BS,))
        assert_shape("marker log likelihood", marker_log_likelihood_expectation.shape, (T, BS))
        return marker_log_likelihood_expectation, time_log_likelihood_expectation
