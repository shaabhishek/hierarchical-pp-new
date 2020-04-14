import torch
from torch import nn
from torch.optim import Adam

from hyperparameters import OptimizerHyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OptimizerLoader:
    def __init__(self, model: nn.Module, hyperparams: OptimizerHyperparams):
        self.optimizer_hyperparams = hyperparams
        self._init_optimizer(model)

    def _init_optimizer(self, model):
        if self.optimizer_hyperparams.l2_weight_decay > 0:
            self.optimizer = Adam(model.parameters(), lr=self.optimizer_hyperparams.lr,
                                  weight_decay=self.optimizer_hyperparams.l2_weight_decay)
        else:
            self.optimizer = Adam(model.parameters(), lr=self.optimizer_hyperparams.lr)

    def _get_optimizer(self):
        return self.optimizer

    @classmethod
    def from_scratch(cls, model, hyperparams):
        optimizer = cls(model, hyperparams)._get_optimizer()
        return optimizer

    @classmethod
    def from_model_checkpoint(cls, model_checkpoint_path, model, hyperparams):
        optimizer = cls(model, hyperparams)._get_optimizer()
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_num = checkpoint['epoch']
        print(f"Loading optimizer from {model_checkpoint_path}, Epoch number: {epoch_num}")
        return optimizer
