import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Exponential

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # intervals = intervals.clamp(0.1, 2.5)
    t = torch.stack([history, intervals], dim=2)
    
    return t

def generate_autoregressive_data(time_step = 100, num_sample = 80, num_clusters=3, debug=False):
    def _alpha_n(interval_history, mu, gamma, mem_vec, m):
        """
            Input:
                interval_history: Tensor of shape num_clusters x n-1
                mu: base duration, Tensor of length num_clusters
                gamma: Tensor of length num_clusters
                mem_vec: Tensor of shape num_clusters x m
                m: number of lookback steps, Scalar
            Output:
                alpha_n: Tensor of shape num_clusters
        """
        _, _, history_size = interval_history.shape
        window_size = min(m, history_size)
        past_effects = interval_history[:,:,-window_size:]*mem_vec[:,-window_size:]
        inverse_alpha = mu + gamma * past_effects.sum(dim=-1)
        return torch.div(1, inverse_alpha)
    
    # effect of previous intervals
    m = 5
    
    # for each cluster, we have different
    # base_mu, gamma, and memory_vector
    vals_base_mu = torch.rand(num_clusters)
    vals_gamma = torch.rand(num_clusters)
    mem_vec = torch.rand(num_clusters, m)
    mem_vec /= mem_vec.sum(dim=-1, keepdim=True)
    
    
    interval_history = torch.zeros(num_sample, num_clusters, 1)
    # for each time, get alpha and using it compute
    # the duration of the interval.
    # Here the intervals are distributed according to the
    # exponential distribution with rate = alpha

    for n in range(time_step):
        rate = _alpha_n(interval_history, vals_base_mu, vals_gamma, mem_vec, m)
        interval_dist = Exponential(rate=rate)
        duration_n = interval_dist.sample()
        interval_history = torch.cat([interval_history, duration_n.view(num_sample, -1, 1)], dim=-1)
    
    # first interval was all zeros for convenience
    interval_history = interval_history[:,:,1:]
    # combine the data from different clusters into one
    interval_history = interval_history.view(-1, time_step)
    # shape = T x N
    interval_history = interval_history.transpose(0,1)
    timeseries = interval_history.cumsum(0)
    # shape = T x N x 2
    t = torch.stack([timeseries, interval_history], dim=-1)
    if debug == False:
        return t
    else:
        info = {
            'vals_base_mu': torch.rand(num_clusters),
            'vals_gamma': torch.rand(num_clusters),
            'mem_vec': torch.rand(num_clusters, m)
            }
        return t, info

def generate_mpp(type='hawkes', time_step = 100, num_sample = 80, marker_dim = 20, num_clusters=3, seed = 1):
    torch.manual_seed(seed)
    if type == 'hawkes':
        # t = generate_hawkes(time_step, num_sample, num_clusters).to(device)
        t = generate_autoregressive_data(time_step, num_sample, num_clusters).to(device)

    markers = torch.randn(time_step, num_sample, marker_dim).to(device)
    data = {'x': markers, 't': t}
    return data, None

def plot_process(timeseries):
    history = np.array(timeseries)
    time = np.linspace(0, history[-1], 1000)
    intensities = [intensity_hawkes(t, history) for t in time]
    plt.plot(time, intensities)
    plt.scatter(history, np.zeros_like(history))