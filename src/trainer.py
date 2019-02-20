import torch
import numpy as np
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam
from rmtpp import rmtpp
from hrmtpp import hrmtpp
from utils_ import generate_mpp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_synthetic_data(time_step = 10, num_sample = 1000, marker_dim = 20):
     marker = np.random.randn(time_step, num_sample, marker_dim)
     points_ = np.random.rand(time_step, num_sample) * 1.
     cum_sum_points =  np.cumsum(points_, axis = 0)
     t = np.stack([cum_sum_points, points_], axis = 2)
     x, t  = marker.tolist(), t.tolist()
     x = torch.tensor(x)
     t = torch.tensor(t)
     data = {'x':x, 't': t}
     return data, None




def train(model, epoch, data, optimizer, batch_size, val_data):
    start = time.time()
    model.train()
    train_loss = 0
    n_train, n_val = len(data['x'][0]), 1.

    optimizer.zero_grad()
    idxs = np.random.permutation(len(data['x']))
    for i in range(0, n_train, batch_size):
        loss, out = model(data['x'][:,i:i+batch_size, :], data['t'][:,i:i+batch_size,:])
        loss.backward()
        train_loss += loss.item()
    optimizer.step()
    end = time.time()
    val_loss = 0.
    if val_data is not None:
        n_val = len(val_data['x'])
        with torch.no_grad():
            val_loss, val_out = model(val_data['x'], val_data['t'])
    out = [i/n_train for i in out]
    val_out = [i/n_val for i in val_out]
    print("Epoch: {}, NLL Loss: {}, Val Loss: {}, Time took: {}".format(epoch, train_loss/n_train,\
     val_loss/n_val, (end-start)))
    print("Train loss Meta Info: ", out)
    print("Val Loss Meta Info: ", val_out)
    #print(model.base_intensity.item(),model.time_influence.item(), model.embed_time.bias[0].item())


def trainer(model, data = None, val_data=None, lr= 1e-3, epoch = 500, batch_size = 100):
    if data == None:
        data, val_data = create_synthetic_data()

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch_number in range(epoch):
        train(model, epoch_number, data, optimizer, batch_size, val_data)
    return model

if __name__ == "__main__":
    model = rmtpp()
    data, _ = generate_mpp()
    val_data, _ = generate_mpp(num_sample = 150)
    #import pdb; pdb.set_trace()
    trainer(model, data, val_data)




