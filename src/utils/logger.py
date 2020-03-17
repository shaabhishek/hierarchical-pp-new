import torch
import numpy as np

class Logger:
    def __init__(self, marker_type:str):
        self.marker_type = marker_type
        self.metrics = ['loss', 'marker_ll', 'time_ll', 'accuracy',  'time_rmse']
        self.logged_metrics = {}
        self.best_epoch = {}
        self.best_valid_loss = {}
        self.improvement_map = {'loss':'small', 'marker_ll':'large', 'time_ll':'large', 'auc':'large','accuracy':'large', 'marker_rmse':'small', 'time_rmse':'small'}

    def init_logged_metrics(self):
        for split in ['train', 'valid']:
            self.logged_metrics[split] = {}
            for metric_name in self.metrics:
                self.logged_metrics[split][metric_name] = {}

        for metric_name in self.metrics:
            self.best_epoch[metric_name] = 1
            self.best_valid_loss[metric_name] = None

    def log_train_epoch(self, epoch_num:int, train_info_dict:dict, valid_info_dict:dict):
        for metric_name in self.metrics:
            self.logged_metrics['train'][metric_name][epoch_num] =  train_info_dict.get(metric_name, float('inf'))
            self.logged_metrics['valid'][metric_name][epoch_num] =  valid_info_dict.get(metric_name, float('inf'))

            if (self.best_valid_loss[metric_name] is None) or \
                (self.improvement_map[metric_name] == 'small' and valid_info_dict[metric_name] < self.best_valid_loss[metric_name]) or \
                (self.improvement_map[metric_name] == 'large' and valid_info_dict[metric_name] > self.best_valid_loss[metric_name]):
                
                self.best_valid_loss[metric_name] = valid_info_dict[metric_name]
                self.best_epoch[metric_name] = epoch_num

    def print_train_epoch(self, epoch_num:int, train_info_dict:dict, valid_info_dict:dict):
        print('epoch', epoch_num + 1)
        if self.marker_type == 'categorical':
            print(f"Validation Accuracy: {valid_info_dict['accuracy']:.3f}, \t\t\t Train Accuracy: {train_info_dict['accuracy']:.3f}")
        else:
            raise NotImplementedError

        print(f"Validation Loss: {valid_info_dict['loss']:.3f}, \t\t\t Train Loss: {train_info_dict['loss']:.3f}")
        print(f"Validation Marker LL: {valid_info_dict['marker_ll']:.3f}, \t\t\t Train Marker LL: {train_info_dict['marker_ll']:.3f}")
        print(f"Validation Time LL: {valid_info_dict['time_ll']:.3f}, \t\t\t Train Time LL: {train_info_dict['time_ll']:.3f}")

def test():
    config = {"marker_type": "categorical"}
    logger = Logger(**config)
    
    dummy_train_info = {'loss': 1.123345, 'time_rmse': 2.345677, 'accuracy': 0.09999, 'auc': 0.2312312,\
        'marker_rmse': 3.213123, 'marker_ll': 1.23131241, 'time_ll': 2.4123123}
    dummy_val_info = {'loss': 1.123345, 'time_rmse': 2.345677, 'accuracy': 0.09999, 'auc': 0.2312312,\
        'marker_rmse': 3.213123, 'marker_ll': 1.23131241, 'time_ll': 2.4123123}
    
    logger.print_train_epoch(42, dummy_train_info, dummy_val_info)
    
if __name__ == "__main__":
    test()