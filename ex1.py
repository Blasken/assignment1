import os
import sys
import common
import numpy as np
import scipy as sp
from scipy.special import erf, erfc
import matplotlib.pyplot as plt

"""
Write a computer program imple-menting the Hopfield model with synchronous
updating (LN1pp. 10 and 14)according to the McCulloch-Pitts rule

Si <- sgn(sum_j(Wij Sj)).

The weights wij are given by Hebb’s rule fori6=j, andwii= 0. Store p random
patterns (LN pp. 16 and 19) and use your computer program tofind how the
one-step error probabilityPErrordepends onp/N. Parameters:

N= 100,200
p= 10,20,30,40,50,75,100,150,200

For each datapoint average over enough independent trials so that you obtain
a precise estimate of PError. Plot your simulation results together with the
correspondingtheoretical curve as a function of p/N. Discuss
"""

def run(steps=10):
    Ns = [100, 200]
    P = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    errors = []

    for N in Ns:
        sub_errors = []
        for p in P:
            subsum = 0 #np.zeros([p])
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
#     plt.legend((n100, n200), ("N=100", "N=200"), scatterpoints=1)
    plt.xlabel("p/N")
    plt.ylabel("P(Error)")
    plt.savefig("ex1")
    plt.show()

def crosstalk_term(patterns):
    dsum = patterns.dot(patterns.t)
    np.fill_diagonal(dsum, 0)
    dsum = patterns.dot(dsum)
    dsum = dsum / len(patterns[0])
    return dsum
