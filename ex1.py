import common
import numpy as np
import scipy as sp
from scipy.special import erf, erfc
import matplotlib.pyplot as plt

def run(steps=1000):
    Ns = [100, 200]
    P = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    errors = []

    for N in Ns:
        sub_errors = []
        for p in P:
            subsum = 0
            for _ in range(steps):
                patterns = common.random_patterns(N, p)
                W = common.hebbs_rule(patterns)
                subsum += np.sum(np.not_equal(patterns, step(W, patterns))) / (p*N)
            sub_errors.append(subsum / steps)
        errors.append(sub_errors)
    _plot(errors, N, P)

def step(W, S):
    return np.sign(W.dot(S.T)).T

def _plot(errors, N, P):
    x1 = [p / 100 for p in P]
    x2 = [p / 200 for p in P]
    p = np.linspace(min(P)/2,max(P))
    Y = erfc(np.sqrt(100/(2*p)))/2
    erf = plt.plot(p/100, Y, 'k-', label="Theoretical")
    n100 = plt.plot(x1, errors[0], color='b', marker='+', linestyle='', label="N=100")
    n200 = plt.plot(x2, errors[1], color='r', marker='x', linestyle='', label="N=200")

    plt.legend(loc=4)
    plt.xlabel("p/N")
    plt.ylabel("P(Error)")
    plt.savefig("ex1")
    plt.show()
