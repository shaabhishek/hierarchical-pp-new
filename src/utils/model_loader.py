import sys

import torch

from parameters import DataModelParams
from hyperparameters import BaseModelHyperparams, RMTPPHyperparams, Model1Hyperparams, Model2Hyperparams, \
    Model2FilterHyperparams, Model2NewHyperparams

sys.path.insert(0, './../')
from rmtpp import RMTPP
from model2 import Model2
from model2_filt import Model2Filter
from model2_new import Model2 as Model2New
from model1 import Model1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoader:
    def __init__(self, model_params: DataModelParams, hyperparams):
        self.model_hyperparams = hyperparams
        self.model = self._load_model(self.model_hyperparams).to(device)

    @staticmethod
    def _load_model(hyperparams):
        if isinstance(hyperparams, RMTPPHyperparams):
            model = RMTPP(marker_type=hyperparams.marker_type, marker_dim=hyperparams.marker_dim,
                          time_dim=hyperparams.time_dim,
                          rnn_hidden_dim=hyperparams.rnn_hidden_dim, x_given_t=hyperparams.x_given_t,
                          base_intensity=hyperparams.base_intensity, time_influence=hyperparams.time_influence,
                          gamma=hyperparams.gamma, time_loss=hyperparams.time_loss, dropout=hyperparams.dropout,
                          latent_dim=None, mc_integration_num_samples=hyperparams.mc_integration_num_samples,
                          n_cluster=None, )
        elif isinstance(hyperparams, Model1Hyperparams):
            model = Model1(marker_type=hyperparams.marker_type, marker_dim=hyperparams.marker_dim,
                           time_dim=hyperparams.time_dim,
                           rnn_hidden_dim=hyperparams.rnn_hidden_dim, n_cluster=hyperparams.n_cluster,
                           latent_dim=hyperparams.latent_dim, x_given_t=hyperparams.x_given_t,
                           base_intensity=hyperparams.base_intensity, time_influence=hyperparams.time_influence,
                           gamma=hyperparams.gamma, time_loss=hyperparams.time_loss, dropout=hyperparams.dropout,
                           mc_integration_num_samples=hyperparams.mc_integration_num_samples,
                           n_samples_posterior=hyperparams.n_samples_posterior)

        elif isinstance(hyperparams, Model2Hyperparams):
            model = Model2(marker_type=hyperparams.marker_type, marker_dim=hyperparams.marker_dim,
                           latent_dim=hyperparams.latent_dim,
                           time_dim=hyperparams.time_dim, rnn_hidden_dim=hyperparams.rnn_hidden_dim,
                           n_cluster=hyperparams.n_cluster,
                           x_given_t=hyperparams.x_given_t, base_intensity=hyperparams.base_intensity,
                           time_influence=hyperparams.time_influence, gamma=hyperparams.gamma,
                           time_loss=hyperparams.time_loss,
                           dropout=hyperparams.dropout)

        elif isinstance(hyperparams, Model2FilterHyperparams):
            model = Model2Filter(n_sample=hyperparams.n_sample, marker_type=hyperparams.marker_type,
                                 marker_dim=hyperparams.marker_dim,
                                 latent_dim=hyperparams.latent_dim, time_dim=hyperparams.time_dim,
                                 rnn_hidden_dim=hyperparams.rnn_hidden_dim, n_cluster=hyperparams.n_cluster,
                                 x_given_t=hyperparams.x_given_t, base_intensity=hyperparams.base_intensity,
                                 time_influence=hyperparams.time_influence, gamma=hyperparams.gamma,
                                 time_loss=hyperparams.time_loss,
                                 dropout=hyperparams.dropout)
        elif isinstance(hyperparams, Model2NewHyperparams):
            model = Model2New(marker_type=hyperparams.marker_type,
                              marker_dim=hyperparams.marker_dim,
                              latent_dim=hyperparams.latent_dim, time_dim=hyperparams.time_dim,
                              rnn_hidden_dim=hyperparams.rnn_hidden_dim, n_cluster=hyperparams.n_cluster,
                              x_given_t=hyperparams.x_given_t, base_intensity=hyperparams.base_intensity,
                              time_influence=hyperparams.time_influence, gamma=hyperparams.gamma,
                              time_loss=hyperparams.time_loss,
                              dropout=hyperparams.dropout, n_samples_posterior=hyperparams.n_samples_posterior)
        else:
            raise ValueError("Did not specify model name correctly")
        return model

    @classmethod
    def from_model_checkpoint(cls, model_params, hyperparams):
        self = cls(model_params, hyperparams)
        model_state_path = model_params.get_model_state_path()
        checkpoint = torch.load(model_state_path, map_location=device)
        epoch_num = checkpoint['epoch']
        print(f"Loading model from {model_state_path}, Epoch number: {epoch_num}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model, epoch_num



# class CheckpointedModelLoader(ModelLoader):
#
#     def __init__(self, model_params: DataModelParams, hyperparams):
#         super(CheckpointedModelLoader, self).__init__(model_params, hyperparams)
#         self.model_state_path = model_params.get_model_state_path()
#         self._load_model_state()
#
#     def _load_model_state(self):
#         checkpoint = torch.load(self.model_state_path, map_location=device)
#         print(f"Loading model from {self.model_state_path}, Epoch number: {checkpoint['epoch']}")
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#
#     def save_model_state(self):
#         raise NotImplementedError

# def save_model(model:torch.nn.Module, optimizer:Optimizer, params:Namespace, loss):
#     state = {
#         'epoch':idx,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }

#     path = os.path.join('model', params.save, params.model, file_name)+'_'+ str(idx+1)
#     torch.save(state, path)
