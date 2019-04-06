import numpy as np
import pickle


data_path = '../data/so_2_train.pkl'

with open(data_path, 'rb') as handle:
    data = pickle.load(handle)

times, x = data['t'], data['x']

print("Time statistics, Number of users", len(times))
d = np.concatenate([y[1:] for y in times], axis = None)
print("Interval stat: Min, Max, mu, std: ", d.min(), d.max(), d.mean(), d.std())

print("x statistics")
ys = np.concatenate([y for y in x], axis = None)
vals, cnts = np.unique(ys, return_counts = True)
print("Number of events: ", np.max(vals)+1)
print("Count Distribution: ", cnts.min(), cnts.max(), cnts.mean(), cnts.std())


