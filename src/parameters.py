import argparse
import os
from argparse import Namespace
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from hyperparameters import ModelHyperparams
from utils.helper import make_intermediate_dirs_if_absent


class TimestampedFilenameMixin:
    setup_time = datetime.now()

    @classmethod
    def get_timestamped_file_name(cls, filename):
        return f"{filename}_{cls.setup_time.strftime('%y_%m_%d_%H_%M_%S')}"


class ModelFileParams:
    def __init__(self, params: Namespace):
        self.identifier_param_dict = OrderedDict([
            ('', params.run_label),
            ('_g', params.gamma), ('_do', params.dropout),
            ('_b', params.batch_size),
            ('_h', params.rnn_hidden_dim), ('_l2', params.l2),
            ('_l', params.latent_dim),
            ('_gn', params.maxgradnorm), ('_lr', params.lr),
            ('_c', params.n_cluster),
            ('_s', params.seed), ('_tl', params.time_loss),
            ('_ai', params.anneal_iter)])

    def get_model_file_identifier(self):
        file_name = ''
        for identifier, param_value in self.identifier_param_dict.items():
            file_name += f"{identifier}{param_value}"
        return file_name


class BaseParams:
    def __init__(self, params: Namespace):
        self.dataset_name = params.data_name
        self.model_name = params.model


class DataParams(BaseParams):
    def __init__(self, params: Namespace, split_name):
        super(DataParams, self).__init__(params)

        # Data-specific parameters
        # Data-location
        self.dataset_dir = params.data_dir
        self.split_name = split_name
        self.cross_val_idx = params.cv_idx
        # Data-type & shape
        self.batch_size = params.batch_size
        self.marker_type = params.marker_type

    def get_data_file_path(self):
        data_file_name = self.dataset_name + "_" + str(self.cross_val_idx) + f"_{self.split_name}.pkl"
        return Path(os.path.join('..', 'data', data_file_name))


class ModelParams(BaseParams):
    def __init__(self, params: Namespace, model_filename):
        super(ModelParams, self).__init__(params)
        self.model_name = params.model
        self.set_model_filename(model_filename)

    @staticmethod
    def _create_model_filename(model_file_identifier):
        """Format: identifier_timestamp.pt"""
        return TimestampedFilenameMixin.get_timestamped_file_name(model_file_identifier) + ".pt"

    def get_model_filename(self):
        return self._model_filename

    def set_model_filename(self, filename):
        self._model_filename = filename

    # def create_and_set_filename(self):
    #     self.set_model_filename(self._create_model_filename())

    @classmethod
    def from_identifier(cls, params, model_file_identifier):
        model_filename = cls._create_model_filename(model_file_identifier)
        return cls(params, model_filename)


class DataModelParams(BaseParams):

    def __init__(self, params, model_filename, split_name="train"):
        super(DataModelParams, self).__init__(params)
        self.data_params = DataParams(params, split_name)
        self.model_params = ModelParams(params, model_filename)
        self.make_model_dir()

    def make_model_dir(self):
        self._model_state_dir = Path(os.path.join('model', self.dataset_name, self.model_name))
        make_intermediate_dirs_if_absent(self._model_state_dir)

    def __getattr__(self, item):
        """Only called when search for attribute within the class fails"""
        if hasattr(self.data_params, item):
            return getattr(self.data_params, item)
        elif hasattr(self.model_params, item):
            return getattr(self.model_params, item)
        else:
            raise AttributeError(f"Did not find the attribute {item} in data or model")

    def get_model_state_path(self, model_filename: str = None):
        if model_filename is not None:
            return self._model_state_dir / model_filename
        elif self._model_filename is not None:
            return self._model_state_dir / self.get_model_filename()
        else:
            raise ValueError("Model state file name (identifier) not specified, or doesn't exist yet")

    @classmethod
    def from_identifier(cls, params, model_file_identifier, split_name):
        self = object.__new__(cls)
        self.data_params = DataParams(params, split_name)
        self.model_params = ModelParams.from_identifier(params, model_file_identifier)
        self.make_model_dir()
        return self


class LoggingParams(BaseParams, TimestampedFilenameMixin):
    def __init__(self, params, model_file_params: ModelFileParams):
        super(LoggingParams, self).__init__(params)
        self.filename = model_file_params.get_model_file_identifier()
        self._logs_dir = Path(os.path.join('result', self.dataset_name, self.model_name))
        make_intermediate_dirs_if_absent(self._logs_dir)

    def get_logs_file_path(self):
        return self._logs_dir / (self.get_timestamped_file_name(self.filename) + ".log")

    def get_tensorboard_log_dir(self):
        return Path(os.path.join('tb_logs', self.dataset_name, self.model_name, self.get_timestamped_file_name("")))


class PredictionParams(BaseParams, TimestampedFilenameMixin):
    def __init__(self, params, model_file_params: ModelFileParams):
        super(PredictionParams, self).__init__(params)
        self.filename = model_file_params.get_model_file_identifier()
        self._predictions_dir = Path(os.path.join('preds', self.dataset_name, self.model_name))
        make_intermediate_dirs_if_absent(self._predictions_dir)

    def get_predictions_file_path(self):
        return self._predictions_dir / self.get_timestamped_file_name(self.filename)


class PlottingParams(BaseParams):
    def __init__(self, params: Namespace):
        super(PlottingParams, self).__init__(params)
        self._plotting_dir = Path(os.path.join('experiments', self.dataset_name, self.model_name))
        make_intermediate_dirs_if_absent(self._plotting_dir)

    def get_plotting_dir(self):
        return self._plotting_dir


class TrainerParams(BaseParams):
    def __init__(self, params: Namespace, data_model_params: DataModelParams,
                 model_hyperparams: ModelHyperparams, model_file_params: ModelFileParams, optimizer_hyperparams):
        super(TrainerParams, self).__init__(params)
        self.data_model_params = data_model_params
        self.model_hyperparams = model_hyperparams
        self.model_file_params = model_file_params
        self.optimizer_hyperparams = optimizer_hyperparams

        self.num_training_iterations = params.max_iter


class TestingParams(BaseParams):
    def __init__(self, params, data_model_params: DataModelParams, model_hyperparams: ModelHyperparams,
                 prediction_params: PredictionParams, model_file_params: ModelFileParams):
        super(TestingParams, self).__init__(params)
        self.data_model_params = data_model_params
        self.model_hyperparams = model_hyperparams
        self.prediction_params = prediction_params
        self.model_file_params = model_file_params

        self.num_training_iterations = params.max_iter


def test_data_model_params():
    from data_model_sandbox import get_argparse_parser_params
    model_name = 'rmtpp'
    dataset_name = 'simulated_hawkes'
    params = get_argparse_parser_params(model_name, dataset_name)
    split_name = 'train'

    # Test 1
    model_filename = 'singletrained_g1_do0.5_b16_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40.pt'
    model_state_path = Path(os.path.join('model', dataset_name, model_name, model_filename))
    data_model_params = DataModelParams(params, model_filename, split_name)

    assert data_model_params.dataset_name == dataset_name
    assert data_model_params.dataset_dir == params.data_dir
    assert data_model_params.get_model_state_path() == model_state_path

    # Test 2
    new_model_filename = "bla"
    new_model_state_path = Path(os.path.join('model', dataset_name, model_name, new_model_filename))
    data_model_params.set_model_filename(new_model_filename)
    assert data_model_params.get_model_state_path() == new_model_state_path


if __name__ == '__main__':
    test_data_model_params()


def _augment_params(params: Namespace):
    params.cv_idx = 1

    ###Fixed parameter###
    if params.data_name == 'mimic2':
        params.marker_dim = 75
        params.base_intensity = -0.
        params.time_influence = 1.
        params.time_dim = 2
        params.marker_type = 'categorical'
        params.batch_size = 32

    elif params.data_name == 'so':
        params.marker_dim = 22
        params.time_dim = 2
        params.base_intensity = -5.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32

    elif params.data_name == 'meme':
        params.marker_dim = 5000
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 64
        params.time_scale = 1e-3


    elif params.data_name == 'retweet':
        params.marker_dim = 3
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32
        params.time_scale = 1e-3

    elif params.data_name == 'book_order':
        params.marker_dim = 2
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 10

    elif params.data_name == 'lastfm':
        params.marker_dim = 3150
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32

    elif params.data_name == 'simulated_hawkes':
        params.marker_dim = 3150
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 16

    elif 'syntheticdata' in params.data_name:
        params.marker_dim = 2
        params.time_dim = 2
        params.base_intensity = 0.
        params.time_influence = 0.01
        params.marker_type = 'categorical'
        params.batch_size = 32


    else:  # different dataset. Encode those details.
        raise ValueError

    if params.time_loss == 'intensity':
        params.n_sample = 1
    if params.time_loss == 'normal':
        params.n_sample = 5

    return params


def setup_parser():
    parser = argparse.ArgumentParser(description='Script to test Marked Point Process.')

    ###Validation Parameter###
    parser.add_argument('--max_iter', type=int, default=1, help='number of iterations')
    parser.add_argument('--anneal_iter', type=int, default=40, help='number of iteration over which anneal goes to 1')
    parser.add_argument('--rnn_hidden_dim', type=int, default=256, help='rnn hidden dim')
    parser.add_argument('--maxgradnorm', type=float, default=10.0, help='maximum gradient norm')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1,
                        help='tradeoff of time and marker in loss. marker loss + gamma * time loss')
    parser.add_argument('--l2', type=float, default=0., help='regularizer with weight decay parameter')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('--latent_dim', type=int, default=20, help='latent dim')
    parser.add_argument('--x_given_t', action='store_true', help='whether x given t')
    parser.add_argument('--n_cluster', type=int, default=10, help='number of cluster')

    ###Helper Parameter###
    parser.add_argument('--model', type=str, default='model2', help='model name')
    parser.add_argument('--time_loss', type=str, default='intensity',
                        help='whether to use normal loss or intensity loss')
    parser.add_argument('--time_scale', type=float, default=1, help='scaling factor to multiply the timestamps with')
    parser.add_argument('--skiptrain', action='store_true', help='disable training if flag is provided')
    parser.add_argument('--skiptest', action='store_true', help='enable testing')
    parser.add_argument('--data_dir', type=str, default='../data/', help='data directory')
    parser.add_argument('--best_epoch', type=int, default=10, help='best epoch')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--dump_cluster', type=int, default=0, help='whether to dump cluster while Testing')
    parser.add_argument('--data_name', type=str, default='mimic2', help='data set name')
    parser.add_argument('--run_label', type=str, default='', help='label for the run')

    # Model Specific HyperParameters
    parser.add_argument('--mc_num_samples', type=int, default=10, help='number of MC samples per event time prediction')

    return parser
