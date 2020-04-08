from torch import nn
from torch.optim import Adam

from hyperparameters import OptimizerHyperparams


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

    def get_optimizer(self):
        return self.optimizer
