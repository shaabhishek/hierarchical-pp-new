import numpy as np
import pickle


data_path = '../data/subreddit_train.pkl'

with open(data_path, 'rb') as handle:
    data = pickle.load(handle)

times, x = data['t'], data['x']

print("Time statistics, Number of users", len(times))
d = np.concatenate([y[1:, 1] for y in times], axis = None)
print("Interval stat: Min, Max, mu, std: ", d.min(), d.max(), d.mean(), d.std())

print("x statistics")
ys = np.concatenate([y for y in x], axis = None)
vals, cnts = np.unique(ys, return_counts = True)
print("Number of reddits: ", len(vals))
print("Count Distribution: ", cnts.min(), cnts.max(), cnts.mean(), cnts.std())
print("Number of reddits greater than 10, 100, 1000, 10000, 100000 is : ", np.sum(cnts>10), np.sum(cnts>100),np.sum(cnts>1000), np.sum(cnts>10000),np.sum(cnts>100000))

