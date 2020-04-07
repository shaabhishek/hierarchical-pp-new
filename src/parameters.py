import os
from argparse import Namespace
from pathlib import Path


class DataParams:
    def __init__(self, params: Namespace):
        self.dataset_dir = params.data_dir
        self.dataset_name = params.data_name
        self.batch_size = params.batch_size
        self.cross_val_idx = params.cv_idx

        # self.time_dim = params.time_dim
        # self.marker_type = params.marker_type
        # self.marker_dim = params.marker_dim

    def get_data_file_path(self, split_name):
        return self.dataset_dir + self.dataset_name + "_" + str(self.cross_val_idx) + f"_{split_name}.pkl"


class ModelParams:
    def __init__(self, params: Namespace, model_file_identifier=None):
        self.model_name = params.model
        self.model_file_identifier = model_file_identifier


class DataModelParams(DataParams, ModelParams):
    def __init__(self, params: Namespace, model_file_identifier=None):
        super().__init__(params, model_file_identifier=None)
        if self.model_file_identifier is not None:
            self._model_state_path = Path(
                os.path.join('model', self.dataset_name, self.model_name, self.model_file_identifier))
        else:
            self._model_state_path = None

    def get_model_state_path(self):
        return self._model_state_path


class PlottingParams(DataModelParams):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self._plotting_dir = Path(os.path.join('experiments', self.dataset_name, self.model_name))

    def get_plotting_dir(self):
        return self._plotting_dir


class ModelHyperparams():
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
