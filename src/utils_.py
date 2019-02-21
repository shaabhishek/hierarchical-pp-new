import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from torch.distributions import Categorical

def intensity_hawkes(t,history, mu=torch.tensor(0.2), alpha=torch.tensor(0.8), beta=torch.tensor(1.)):
    # intensity = mu + alpha*sum(torch.exp(-beta*(t - ti)) for ti in history if ti <= t)
    time_diffs = t - torch.tensor(history, dtype=torch.float)
    intensity = mu + alpha * torch.sum(torch.exp(-beta * time_diffs[time_diffs > 0]))
    return intensity.item()


def homo_poisson(mu, last_time):
    u = random.random()
    return last_time - np.log(1-u)/mu 

def hawkes(intensity_fn, time_step=100):
    history = [0]
    
    for point_idx in range(time_step):
        intensity_max = intensity_fn(history[-1], history)
        t = history[-1]
        while True:
            t = homo_poisson(intensity_max, t)
            u = random.random()
            if u <= intensity_fn(t, history)/intensity_max and (t - history[-1])>1e-1:
                history.append(t)
                break
    return history

def get_intervals(timeseries, dim=0):
    """
        dim: axis of time
    """
    if dim != 0:
        timeseries = timeseries.transpose(0,dim)
    shifted = torch.cat([torch.zeros_like(timeseries[0:1]), timeseries[:-1]], dim=0)
    intervals = timeseries - shifted
    if dim != 0:
        intevals = intevals.transpose(0,dim)
    return intervals

def generate_hawkes(time_step, num_sample, num_clusters):
    history = [[0] for _ in range(num_sample)]

    vals_mu = torch.rand(num_clusters)
    prob_mu = torch.rand(num_clusters)
    prob_mu /= prob_mu.sum()
    mu_dist = Categorical(probs=prob_mu)
    
    for sample_idx in range(num_sample):
        mu = vals_mu[mu_dist.sample()]
        intensity_fn = lambda t, history: intensity_hawkes(t, history, mu=mu)
        timeseries = hawkes(intensity_fn, time_step)
        history[sample_idx] = timeseries

    history = torch.tensor(history).transpose(0,1)[1:]
    intervals = get_intervals(history, dim=0)
    intervals = intervals.clamp(0.1, 2.5)
    t = torch.stack([history, intervals], dim=2)
    
    return t

def generate_mpp(type='hawkes', time_step = 100, num_sample = 80, marker_dim = 20, num_clusters=3, seed = 1):
    torch.manual_seed(seed)
    if type == 'hawkes':
        t = generate_hawkes(time_step, num_sample, num_clusters)

    markers = torch.randn(time_step, num_sample, marker_dim)
    data = {'x': markers, 't': t}
    return data, None

def plot_process(timeseries):
    history = np.array(timeseries)
    time = np.linspace(0, history[-1], 1000)
    intensities = [intensity_hawkes(t, history) for t in time]
    plt.plot(time, intensities)
    plt.scatter(history, np.zeros_like(history))