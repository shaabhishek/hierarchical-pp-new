import torch

from hyperparameters import RMTPPHyperParams, Model1HyperParams, Model2HyperParams
from model1 import Model1
from model2_new import Model2
from rmtpp import RMTPP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoader:
    def __init__(self, hyperparams):
        self.model_hyperparams = hyperparams
        self.model = self._load_model(self.model_hyperparams).to(device)

    @staticmethod
    def _load_model(hyperparams):
        if isinstance(hyperparams, RMTPPHyperParams):
            model = RMTPP(hyperparams)
        elif isinstance(hyperparams, Model1HyperParams):
            model = Model1.from_model_hyperparams(hyperparams)
        elif isinstance(hyperparams, Model2HyperParams):
            model = Model2.from_model_hyperparams(hyperparams)
        # elif isinstance(hyperparams, Model2FilterHyperParams):
        #     model = Model2Filter.from_model_hyperparams(hyperparams)
        else:
            raise ValueError("Did not specify model name correctly")
        return model

    @classmethod
    def from_model_checkpoint(cls, model_params):
        model_state_path = model_params.get_model_state_path()
        checkpoint = torch.load(model_state_path, map_location=device)
        hyperparams = checkpoint['model_hyperparams']
        epoch_num = checkpoint['epoch']
        print(f"Loading model from {model_state_path}, Epoch number: {epoch_num}")
        self = cls(hyperparams)
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
