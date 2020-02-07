import numpy as np
from scipy.signal import lfilter



def discount(x, gamma, window=None):
    if window is None:
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        gammas = (gamma*np.ones(window))**np.arange(window)
        y = [np.dot(x[i:i+window], gammas) for i in range(len(x)-window)]
        y.extend(window*np.array(x[-window:]))
        return np.asarray(y)


def extract_sars(rollout):
    s0 = ([np.squeeze(_[0]) for _ in rollout])
    a = ([np.squeeze(_[1]) for _ in rollout])
    r = ([_[2] for _ in rollout])
    s1 = ([np.squeeze(_[3]) for _ in rollout])
    return (s0, a, r, s1)
