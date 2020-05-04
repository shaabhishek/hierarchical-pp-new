import numpy as np
import torch
from torch.utils.data import DataLoader

from base_model import _get_timestamps_and_intervals_from_data
from hyperparameters import HawkesHyperparams, BaseModelHyperParams
from model1 import Model1
from parameters import DataModelParams, DataParams, _augment_params, setup_parser
from rmtpp import RMTPP
from utils.data_loader import get_dataloader
from utils.model_loader import ModelLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HawkesModel:
    def __init__(self, model_hyperparams: HawkesHyperparams):
        self.lambda_0 = model_hyperparams.lambda_0
        self.alpha = model_hyperparams.alpha
        self.sigma = model_hyperparams.sigma

    def get_intensity(self, t: float, data_timestamps: np.ndarray):
        # params: [lambda0, alpha, sigma]
        # data_timestamps must be numpy array
        timesteps_in_past = data_timestamps[(data_timestamps < t)]
        intensity = self.lambda_0 + self.alpha * np.sum(np.exp(-1. * (t - timesteps_in_past) / self.sigma))
        return intensity


class HawkesProcessDataModelSandBox:
    def __init__(self, data_params: DataParams, model_hyperparams: HawkesHyperparams):
        # super(HawkesProcessDataModelSandBox, self).__init__(data_params)
        self.dataloader = load_dataloader_from_params(data_params)
        self.model = HawkesModel(model_hyperparams)

    def setup(self, idx: int = 1):
        _, t_data, _ = self.dataloader.collate_fn([self.dataloader.dataset[idx]])  # (T, BS=1, 1)
        data_timestamps = t_data[:, :, 1:2].cpu().numpy().flatten()
        return data_timestamps

    def get_intensity_over_grid(self, data_timestamps, grid_size=None):
        if grid_size is not None:
            grid_times = np.linspace(0, data_timestamps[-1], grid_size)
        else:
            grid_times = data_timestamps
        intensity = np.array([self.model.get_intensity(t=_t, data_timestamps=data_timestamps) for _t in grid_times])
        return intensity, grid_times


class BaseNNDataModelSandBox:
    def __init__(self, data_model_params: DataModelParams):
        self.dataloader = load_dataloader_from_params(data_model_params)
        self.model = _load_model_from_params(data_model_params)

    def _setup(self, sequence_idx):
        raise NotImplementedError

    def get_intensity_over_grid(self, sequence_idx, grid_size):
        raise NotImplementedError

    def _get_sequence_tensors(self, sequence_idx):
        x_data, t_data, mask = self.dataloader.collate_fn([self.dataloader.dataset[sequence_idx]])
        return t_data, x_data, mask

    @staticmethod
    def _get_grid_timestamps_from_grid_size(data_timestamps, grid_size):
        if grid_size is not None:
            sequence_durations, _ = data_timestamps.max(dim=0)  # (BS,1)
            grid_times = torch.stack([torch.linspace(0, t, grid_size) for t in sequence_durations.squeeze(-1)],
                                     dim=1).to(device)  # (N, BS)
        else:
            grid_times = data_timestamps.squeeze(-1)
        return grid_times


class RMTPPDataModelSandBox(BaseNNDataModelSandBox):
    model: RMTPP

    def get_intensity_over_grid(self, sequence_idx: int, grid_size: int):
        hidden_seq, data_timestamps = self._setup(sequence_idx)  # (T, BS=1, h_dim), (T, BS=1, 1)

        log_intensity, evaluated_timestamps = self._transform_intensity_over_grid(hidden_seq, data_timestamps,
                                                                                  grid_size=grid_size)
        return log_intensity, data_timestamps, evaluated_timestamps

    def _setup(self, sequence_idx: int = 0):
        t_data, x_data, _ = self._get_sequence_tensors(sequence_idx)
        _, data_timestamps = _get_timestamps_and_intervals_from_data(t_data)  # (T, BS=1, 1)
        with torch.no_grad():
            hidden_seq, _, _ = self.model.get_hidden_states_from_input(x_data, t_data)
        return hidden_seq, data_timestamps

    def _transform_intensity_over_grid(self, hidden_seq, data_timestamps, grid_size=1000):
        evaluated_timestamps = self._get_grid_timestamps_from_grid_size(data_timestamps, grid_size)
        log_intensity = self.model.marked_point_process_net.get_intensity_over_timestamps(
            hidden_seq, data_timestamps, evaluated_timestamps)
        return log_intensity, evaluated_timestamps


class Model1DataModelSandBox(BaseNNDataModelSandBox):
    model: Model1

    def get_intensity_over_grid(self, sequence_idx: int, grid_size: int):
        hidden_seq, data_timestamps = self._setup(sequence_idx)  # (T, BS=1, h_dim), (T, BS=1, 1)
        log_intensity, evaluated_timestamps = self._transform_intensity_over_grid(hidden_seq, data_timestamps, grid_size)
        return log_intensity, data_timestamps, evaluated_timestamps

    def _setup(self, sequence_idx: int = 0):
        t_data, x_data, mask = self._get_sequence_tensors(sequence_idx)
        _, data_timestamps = _get_timestamps_and_intervals_from_data(t_data)  # (T, BS=1, 1)
        with torch.no_grad():
            phi_x, phi_xt = self.model.preprocess_inputs(x_data, t_data)
            augmented_hidden_seq, (posterior_sample_y, posterior_dist_y), (
                posterior_sample_z, posterior_dist_z) = self.model.encoder(phi_xt, 0.001, mask)
            phi_hzy_seq = self.model.decoder.preprocess_latent_states(augmented_hidden_seq, posterior_sample_y,
                                                                      posterior_sample_z, None, None)
        return phi_hzy_seq, data_timestamps

    def _transform_intensity_over_grid(self, phi_hzy_seq, data_timestamps, grid_size):
        evaluated_timestamps = self._get_grid_timestamps_from_grid_size(data_timestamps, grid_size)
        log_intensity = self.model.decoder.marked_point_process_net.get_intensity_over_timestamps(
            phi_hzy_seq, data_timestamps, evaluated_timestamps)
        return log_intensity, evaluated_timestamps


def _load_model_from_params(data_model_params: DataModelParams):
    model, _ = ModelLoader.from_model_checkpoint(data_model_params)
    model = model.to(device)
    return model


def load_dataloader_from_params(data_model_params: [DataModelParams, DataParams]):
    # Data should reside in this path for all datasets. Ideally 5 cross fold validation.
    data_path = data_model_params.get_data_file_path()
    dataloader: DataLoader = get_dataloader(data_path, data_model_params.marker_type, data_model_params.batch_size)
    return dataloader


def get_argparse_parser_params(model_name=None, dataset_name=None):
    parser = setup_parser()
    params = parser.parse_args()
    ###
    params.model = model_name if model_name is not None else params.model
    params.data_name = dataset_name if dataset_name is not None else params.data_name
    ###
    _augment_params(params)
    return params
