import random
import numpy as np

def intensityHP(t, hist, params):
    # params: {lambda0, alpha, beta}
    # hist must be numpy array
    hist = hist[(hist < t)]
    return params[0] + params[1] * np.sum(np.exp( -1.* (t - hist)/params[2] ))

def genHP(T, params):
    t = 0
    hist = []
    while (t < T):
        upperbnd = intensityHP(t, np.array(hist), params)
        increment = np.random.exponential(1/upperbnd)
        t += increment
        u = random.random()*increment
        if ((t<T) and (u <= intensityHP(t, np.array(hist), params))):
            hist.append(t)
    return {'t': hist, 'x': list(np.ones_like(hist))}

def intensitySCP(t, hist, params):
    # params: {mu, alpha}
    # hist must be numpy array
    hist = hist[(hist < t)]
    return np.exp(params[0]*t - np.sum(len(hist)*params[1]))

def genSCP(T, params):
    t = 0
    hist = []
    while (t < T):
        upperbnd = intensitySCP(t+1, np.array(hist), params)
        increment = np.random.exponential(1/upperbnd)
        t += increment
        u = random.random()*increment
        if ((t<T) and (u <= intensitySCP(t, np.array(hist), params))):
            hist.append(t)
    return {'t': hist, 'x': list(np.ones_like(hist))}

def genHomPP(T, params):
    # params = [lambda]
    t = 0
    hist = []
    while (t < T):
        increment = np.random.exponential(1/params[0])
        t += increment
        if ((t<T)):
            hist.append(t)
    return {'t': hist, 'x': list(np.ones_like(hist))}

def intensityNonHomPP(t, params):
    # params: {alpha}
    return (0.1*np.sin(np.pi*t) + 0.3*np.cos(.4 * np.pi*t) + 1)*.08

def genNonHomPP(T, params):
    t = 0
    hist = []
    while (t < T):
        upperbnd = 2
        increment = np.random.exponential(1/upperbnd)
        t += increment
        u = random.random()*increment
        if ((t<T) and (u <= intensityNonHomPP(t, params))):
            hist.append(t)
    return {'t': hist, 'x': list(np.ones_like(hist))}

def intensitySSCT(r, y, params):
    # params: [a1...am]
    return np.exp(np.dot(params, r*y))

def genpointSSCT(r, y, params):
    intensity = intensitySSCT(np.array(r), np.array(y), params)
    increment = np.random.exponential(1/intensity)
    return increment

def markerprobSSCT(k, r, y, params):
    # k is the config of marker : +/- 1
    # all others must be np arrays
    kernel = lambda x: np.exp(x * np.dot(params, r*y))
    return kernel(1)/(kernel(1) + kernel(-1))

def genmarkerSSCT(r, y, params):
    prob_1 = markerprobSSCT(1, np.array(r), np.array(y), np.array(params))
    u = random.random()
    return 1 if (u < prob_1) else -1

def genSSCT(T, params):
    m = 3
    t = 0
    r_hist = [random.random() for _ in range(m)]
    y_hist = list(np.random.randint(0, 2, m)*2 - 1)
    hist = []
    while (t<T):
        increment = genpointSSCT(r_hist[-m:], y_hist[-m:], params)
        t += increment
        rn = int(t%24 < 12)
        yn = genmarkerSSCT(r_hist[-m:], y_hist[-m:], params)
        if (t<T):
            hist.append(t)
            r_hist.append(rn)
            y_hist.append(yn)
    y_hist = list(map(lambda x: x if x==1 else 0, y_hist))
    return {'x': y_hist[m:], 't': hist}

def generate_data(nperproc, T, fn_param_pairs):
    # nperproc = n//len(fn_param_pairs)
    data = {'x': [], 't': []}
    for fn, params in fn_param_pairs:
        for i in range(nperproc):
            len_check = False
            while not len_check:
                data_i = fn(T, params)
                # make sure data is at least length 2
                if (len(data_i['x']) >= 2):
                    len_check = True
            # Generate intervals
            intervals = np.diff([0]+data_i['t'])
            data_i['t'] = np.stack([intervals, data_i['t']]).T
            data_i['x'] = np.array(data_i['x'])
            data['t'].append(data_i['t'])
            data['x'].append(data_i['x'])
            
    return data

def paramsHP():
    alpha = random.random()
    #unif(.2, 1)
    lambda0 = .2 + random.random()*.8 #np.clip(random.random(), 0.2, 1)
    beta = alpha + random.random()*(1-alpha) #np.clip(random.random(), alpha+.1, 1)
    return [lambda0, alpha, beta]

def paramsNonHomPP():
    #unif(0.2, 1)
    lambd = .2 + random.random()*.8
    return [lambd]

def paramsHomPP():
    #unif(0.1, 1)
    lambd = .1 + random.random()*.9
    return [lambd]

def paramsSCP():
    mu = 0.5 + random.random() #unif(0.5, 1.5)
    alpha = .1 + random.random()*.4 #unif(0.1, .5)
    return [mu, alpha]

def generate_synthethic_data_wrapper(nperproc=100, nclus=2, T=100, shuffle=False):
    fns_proc = [genHomPP, genNonHomPP, genHP, genSCP]
    params_proc = [paramsHomPP, paramsNonHomPP, paramsHP, paramsSCP]
    fn_param_pairs = []
    # nperproc = 200

    for proc in range(4):
        for _ in range(nclus):
            params = params_proc[proc]()
            fn_param_pairs.append((fns_proc[proc], params))

    data = generate_data(nperproc, T, fn_param_pairs)
    if shuffle:
        idxs = np.arange(len(data['x']))
        np.random.shuffle(idxs)
        data['x'] = [data['x'][i] for i in idxs]
        data['t'] = [data['t'][i] for i in idxs]

    print("Count:", len(data['x']))
    print("Length stats (min,max,mean,median):",np.min(list(map(len, data['x']))), np.max(list(map(len, data['x']))), np.mean(list(map(len, data['x']))), np.median(list(map(len, data['x']))))    
    return data

if __name__ == "__main__":
    T=25
    nperproc = 50
    nclus = 2
    data = generate_synthethic_data_wrapper(nperproc=nperproc,nclus=nclus, T=T)