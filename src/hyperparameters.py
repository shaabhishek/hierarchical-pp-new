from argparse import Namespace


class HawkesHyperparams:
    def __init__(self, lambda_0=0.2, alpha=0.8, sigma=1.0):
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.sigma = sigma


class OptimizerHyperparams:
    def __init__(self, params: Namespace):
        self.lr = params.lr
        self.l2_weight_decay = params.l2


class BaseModelHyperParams:
    def __init__(self, params: Namespace, **kwargs):
        # Data-specific information
        self.time_dim = params.time_dim
        self.marker_type = params.marker_type
        self.marker_dim = params.marker_dim

        self.rnn_hidden_dim = params.rnn_hidden_dim
        self.x_given_t = params.x_given_t
        self.base_intensity = params.base_intensity
        self.time_influence = params.time_influence
        self.integration_degree = params.integration_degree
        self.gamma = params.gamma
        self.time_loss = params.time_loss
        self.dropout = params.dropout

        self.total_anneal_epochs = params.anneal_iter
        self.grad_max_norm = params.maxgradnorm

        # Preprocessing network dims
        self.time_embedding_net_dims, self.marker_embedding_net_dims, self.emb_dim = self._get_input_preprocessing_net_params()

    def _get_input_preprocessing_net_params(self):
        x_embedding_dims = [128]
        t_embedding_dims = [8]
        time_embedding_net_dims = [self.time_dim, *t_embedding_dims]
        marker_embedding_net_dims = [self.marker_dim, *x_embedding_dims]
        embedding_dim = x_embedding_dims[-1] + t_embedding_dims[-1]
        return time_embedding_net_dims, marker_embedding_net_dims, embedding_dim

    @classmethod
    def from_params(cls, params):
        model_str_class_map = {'rmtpp': RMTPPHyperParams, 'model1': Model1HyperParams, 'model2': Model2HyperParams,
                               'model2_filt': Model2FilterHyperParams, 'model2_new': Model2NewHyperParams}
        model_name = params.model
        return model_str_class_map[model_name](params)

    @property
    def model_name(self):
        raise NotImplementedError


class RMTPPHyperParams(BaseModelHyperParams):
    model_name = 'rmtpp'

    def __init__(self, params: Namespace):
        super(RMTPPHyperParams, self).__init__(params)


class Model1HyperParams(BaseModelHyperParams):
    model_name = 'model1'

    def __init__(self, params: Namespace):
        super(Model1HyperParams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.cluster_dim = params.cluster_dim
        self.n_samples_posterior = params.n_samples_posterior


class Model2HyperParams(BaseModelHyperParams):
    model_name = 'model2'

    def __init__(self, params: Namespace):
        super(Model2HyperParams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.cluster_dim = params.cluster_dim
        self.n_samples_posterior = params.n_samples_posterior


class Model2FilterHyperParams(BaseModelHyperParams):
    model_name = 'model2_filt'

    def __init__(self, params: Namespace):
        super(Model2FilterHyperParams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.cluster_dim = params.cluster_dim
        self.n_samples_posterior = params.n_samples_posterior


class Model2NewHyperParams(BaseModelHyperParams):
    model_name = 'model2_new'

    def __init__(self, params: Namespace):
        super(Model2NewHyperParams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.cluster_dim = params.cluster_dim
        self.n_samples_posterior = params.n_samples_posterior


class EncoderHyperParams:
    def __init__(self, model_hyperparams):
        self.num_posterior_samples = model_hyperparams.n_samples_posterior

        # Inference network hidden dimensions
        encoder_z_hidden_dims = [64, 64]
        encoder_y_hidden_dims = [64]
        z_input_dim = model_hyperparams.rnn_hidden_dim + model_hyperparams.emb_dim + model_hyperparams.cluster_dim

        # Network layer-wise dimensions
        self.rnn_dims = [model_hyperparams.emb_dim, model_hyperparams.rnn_hidden_dim]
        self.y_dims = [model_hyperparams.rnn_hidden_dim, *encoder_y_hidden_dims, model_hyperparams.cluster_dim]
        self.z_dims = [z_input_dim, *encoder_z_hidden_dims, model_hyperparams.latent_dim]


class Encoder1HyperParams(EncoderHyperParams):
    pass


class Encoder2HyperParams(EncoderHyperParams):
    def __init__(self, model_hyperparams):
        super(Encoder2HyperParams, self).__init__(model_hyperparams)
        self.reverse_rnn_dims = [
            model_hyperparams.emb_dim + model_hyperparams.rnn_hidden_dim,
            model_hyperparams.rnn_hidden_dim
        ]  # phi_xt+h -> h


class DecoderHyperParams:
    def __init__(self, model_hyperparams):
        self.marker_type = model_hyperparams.marker_type
        self.time_loss = model_hyperparams.time_loss

    @staticmethod
    def _get_filtering_preprocessing_params(cluster_dim, latent_dim, rnn_hidden_dim):
        filtering_out_dims = [64]
        decoder_in_dim = rnn_hidden_dim + cluster_dim + latent_dim
        return filtering_out_dims[-1], [decoder_in_dim, *filtering_out_dims]

    @staticmethod
    def _get_mpp_config(hidden_rep_dim, model_hyperparams):
        assert isinstance(model_hyperparams, BaseModelHyperParams)
        mpp_config = {
            "input_dim": hidden_rep_dim,
            "marker_dim": model_hyperparams.marker_dim,
            "marker_type": model_hyperparams.marker_type,
            "init_base_intensity": model_hyperparams.base_intensity,
            "init_time_influence": model_hyperparams.time_influence,
            "x_given_t": model_hyperparams.x_given_t,
            "integration_degree": model_hyperparams.integration_degree
        }
        return mpp_config


class Decoder1HyperParams(DecoderHyperParams):
    def __init__(self, model_hyperparams: Model1HyperParams):
        super(Decoder1HyperParams, self).__init__(model_hyperparams)

        filtering_out_dim, self.filtering_preprocessing_module_dims = self._get_filtering_preprocessing_params(
            model_hyperparams.cluster_dim, model_hyperparams.latent_dim, model_hyperparams.rnn_hidden_dim)

        self.marked_point_process_params = self._get_mpp_config(filtering_out_dim, model_hyperparams)


class Decoder2HyperParams(DecoderHyperParams):
    def __init__(self, model_hyperparams: Model2HyperParams):
        super(Decoder2HyperParams, self).__init__(model_hyperparams)
        self.is_smoothing = True
        filtering_out_dim, self.filtering_preprocessing_module_dims = self._get_filtering_preprocessing_params(
            model_hyperparams.cluster_dim, model_hyperparams.latent_dim, model_hyperparams.rnn_hidden_dim)
        smoothing_out_dim, self.smoothing_preprocessing_module_dims = self._get_smoothing_preprocessing_params(
            model_hyperparams.rnn_hidden_dim)
        self.marked_point_process_params = self._get_mpp_config(filtering_out_dim + smoothing_out_dim,
                                                                model_hyperparams)

    def _get_smoothing_preprocessing_params(self, rnn_hidden_dim):
        if self.is_smoothing:
            smoothing_out_dims = [64]
            return smoothing_out_dims[-1], [rnn_hidden_dim, *smoothing_out_dims]
        return 0, None
