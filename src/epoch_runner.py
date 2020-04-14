import math

import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim
from torch.utils.data import DataLoader

from base_model import BaseModel
from epoch_metrics import EpochMetrics
from utils.helper import ProgressBar
from utils.logger import Logger

anneal_model = {'hrmtpp', 'model11', 'model2'}


class EpochRunnerMixin:
    def run_epoch(self, epoch_num) -> dict:
        assert isinstance(self, (TrainEpochRunner, TestEpochRunner, ValidEpochRunner))
        self.set_model_mode()
        self.reset_epoch_metrics()
        num_batches = self.get_num_batches()
        bar = ProgressBar(self.split_name, max=num_batches)
        for b_idx, (input_x, input_t, input_mask) in enumerate(self.dataloader):
            bar.update(b_idx)
            self.batch_start_hook()
            loss, meta_info = self.run_network(input_x, input_t, input_mask, epoch_num)
            self.epoch_metrics.update_batch_metrics(meta_info)
            self.batch_end_hook(loss, epoch_num)
        bar.finish()
        epoch_metrics = self.epoch_metrics.get_reduced_metrics(
            len(self.dataloader.dataset))
        return epoch_metrics


class BaseEpochRunner:
    def __init__(self, model, dataloader: DataLoader, logger: Logger):
        self.model = model
        self.model_name = model.model_name
        self.dataloader = dataloader
        self.logger = logger

    def reset_epoch_metrics(self):
        self.epoch_metrics = EpochMetrics(self.dataloader.marker_type)

    def get_num_batches(self):
        return int(math.ceil(len(self.dataloader.dataset) / self.dataloader.batch_size))


class TrainEpochRunner(BaseEpochRunner, EpochRunnerMixin):
    split_name = "train"

    def __init__(self, model: BaseModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer, logger,
                 total_anneal_epochs: int, grad_max_norm: float):
        super(TrainEpochRunner, self).__init__(model, dataloader, logger)

        self.optimizer = optimizer
        self.total_anneal_epochs = total_anneal_epochs
        self.grad_max_norm = grad_max_norm

    def run_network(self, input_x, input_t, input_mask, epoch_num):
        if self.model_name in anneal_model:
            annealing_value = self.get_kl_annealing_value(epoch_num)
            loss, meta_info = self.model(input_x, input_t, mask=input_mask, anneal=annealing_value)
        else:
            loss, meta_info = self.model(input_x, input_t, mask=input_mask)
        return loss, meta_info

    def batch_start_hook(self):
        self.optimizer.zero_grad()

    def batch_end_hook(self, loss, epoch_num):
        loss.backward()
        if self.grad_max_norm > 0:
            _ = clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)

        # grad_norms_dict = {k: p.grad.norm() for k, p in dict(self.model.named_parameters()).items()}
        # self.logger.writer.add_scalars("grad_norms", grad_norms_dict, epoch_num)
        #
        # params_norms_dict = {k: p.norm() for k, p in dict(self.model.named_parameters()).items()}
        # self.logger.writer.add_scalars("param_norms", params_norms_dict, epoch_num)

        for param_name, param_tensor in dict(self.model.named_parameters()).items():
            self.logger.writer.add_histogram("parameters/" + param_name.replace('.', '/'), param_tensor.grad, epoch_num)
            self.logger.writer.add_histogram("gradients/" + param_name.replace('.', '/'), param_tensor, epoch_num)

        self.optimizer.step()

    def set_model_mode(self):
        self.model.train()

    def get_kl_annealing_value(self, epoch_num):
        return min(1., epoch_num / (self.total_anneal_epochs + 0.))


class EvalEpochRunner(BaseEpochRunner, EpochRunnerMixin):
    def __init__(self, model: BaseModel, dataloader: DataLoader, logger):
        super(EvalEpochRunner, self).__init__(model, dataloader, logger)

    def set_model_mode(self):
        self.model.eval()

    def run_network(self, input_x, input_t, input_mask, epoch_num):
        with torch.no_grad():
            loss, meta_info = self.model(input_x, input_t, mask=input_mask)
        return loss, meta_info

    def batch_start_hook(self, *args, **kwargs):
        pass

    def batch_end_hook(self, *args, **kwargs):
        pass


class TestEpochRunner(EvalEpochRunner):
    split_name = "test"


class ValidEpochRunner(TestEpochRunner):
    split_name = "valid"
