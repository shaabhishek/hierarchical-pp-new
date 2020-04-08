from argparse import Namespace


class HawkesHyperparams:
    def __init__(self, lambda_0=0.2, alpha=0.8, beta=1.0):
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.beta = beta


class BaseModelHyperparams:
    def __init__(self, params: Namespace, **kwargs):
        # Data-specific information
        self.time_dim = params.time_dim
        self.marker_type = params.marker_type
        self.marker_dim = params.marker_dim

        self.rnn_hidden_dim = params.rnn_hidden_dim
        self.x_given_t = params.x_given_t
        self.base_intensity = params.base_intensity
        self.time_influence = params.time_influence
        self.gamma = params.gamma
        self.time_loss = params.time_loss
        self.dropout = params.dropout

        self.total_anneal_epochs = params.anneal_iter
        self.grad_max_norm = params.maxgradnorm

    @property
    def model_name(self):
        raise NotImplementedError


class RMTPPHyperparams(BaseModelHyperparams):
    model_name = 'rmtpp'


class Model1Hyperparams(BaseModelHyperparams):
    model_name = 'model1'

    def __init__(self, params: Namespace):
        super(Model1Hyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster


class Model2Hyperparams(BaseModelHyperparams):
    model_name = 'model2'

    def __init__(self, params: Namespace):
        super(Model2Hyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster


class Model2FilterHyperparams(BaseModelHyperparams):
    model_name = 'model2_filt'

    def __init__(self, params: Namespace):
        super(Model2FilterHyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster
        self.n_sample = params.n_sample


class Model2NewHyperparams(BaseModelHyperparams):
    model_name = 'model2_new'

    def __init__(self, params: Namespace):
        super(Model2NewHyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster
        self.n_sample = params.n_sample


class ModelHyperparams:
    model_str_class_map = {'rmtpp': RMTPPHyperparams, 'model1': Model1Hyperparams, 'model2': Model2Hyperparams,
                           'model2_filt': Model2FilterHyperparams, 'model2_new': Model2NewHyperparams}

    def __new__(cls, params: Namespace):
        model_name = params.model
        return cls.model_str_class_map[model_name](params)


class OptimizerHyperparams:
    def __init__(self, params: Namespace):
        self.lr = params.lr
        self.l2_weight_decay = params.l2


