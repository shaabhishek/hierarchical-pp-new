import os
from argparse import Namespace
from pathlib import Path


class DataParams:
    def __init__(self, params: Namespace,  split_name="train"):
        # Data-specific parameters
        # Data-location
        self.dataset_dir = params.data_dir
        self.dataset_name = params.data_name
        self.split_name = split_name
        self.cross_val_idx = params.cv_idx
        # Data-type & shape
        self.batch_size = params.batch_size
        self.marker_type = params.marker_type

    def get_data_file_path(self):
        data_file_name = self.dataset_name + "_" + str(self.cross_val_idx) + f"_{self.split_name}.pkl"
        return Path(os.path.join('..', 'data', data_file_name))


class DataModelParams(DataParams):
    def __init__(self, params: Namespace, model_file_identifier=None, split_name="train"):
        super(DataModelParams, self).__init__(params, split_name)

        # Model-specific parameters
        self.model_name = params.model
        self.model_file_identifier = model_file_identifier

        if self.model_file_identifier is not None:
            self._model_state_path = self.get_data_file_path()
        else:
            self._model_state_path = None

    def get_model_state_path(self):
        return Path(os.path.join('model', self.dataset_name, self.model_name, self.model_file_identifier))


class PlottingParams:
    def __init__(self, params: Namespace):
        self.dataset_name = params.data_name
        self.model_name = params.model
        self._plotting_dir = Path(os.path.join('experiments', self.dataset_name, self.model_name))

    def get_plotting_dir(self):
        return self._plotting_dir


class HawkesHyperparams:
    def __init__(self, lambda_0=0.2, alpha=0.8, beta=1.0):
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.beta = beta


class ModelHyperparams:
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


class RMTPPHyperparams(ModelHyperparams):
    model_name = 'rmtpp'


class Model1Hyperparams(ModelHyperparams):
    model_name = 'model1'

    def __init__(self, params: Namespace):
        super(Model1Hyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster


class Model2Hyperparams(ModelHyperparams):
    model_name = 'model2'

    def __init__(self, params: Namespace):
        super(Model2Hyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster


class Model2FilterHyperparams(ModelHyperparams):
    model_name = 'model2_filt'

    def __init__(self, params: Namespace):
        super(Model2FilterHyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster
        self.n_sample = params.n_sample


class Model2NewHyperparams(ModelHyperparams):
    model_name = 'model2_new'

    def __init__(self, params: Namespace):
        super(Model2NewHyperparams, self).__init__(params)
        self.latent_dim = params.latent_dim
        self.n_cluster = params.n_cluster
        self.n_sample = params.n_sample


