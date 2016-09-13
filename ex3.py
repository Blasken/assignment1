import os
import sys
import common
import numpy as np
import matplotlib.pyplot as plt

"""
Stochastic Hopfield model. Write a computer program implementing the Hopfield
model with stochastic updating (LN pp. 36). Use Hebb’s rule and take w ii = 0.
Use random patterns (LN pp. 16 and 19).

a) For β = 2 determine the steady-state order parameter m 1 as a function of
α = p/N for N = 500. Discard data points corresponding to the initial transient
of the stochastic dynamics. Average over independent realisations. Plot the
order parameter as a function of α in the range 0 < α ≤ 1. Discuss your results.
Refer to the phase diagram (LN p. 60).( 1p ).

b) Repeat the above for N = 50 , 100 , and 250 choosing p so that α is in the
same range as above. Discuss how the value of N influences the order parameter m1 .
"""


def run():
    N = 500
    alphas = np.linspace(0.01,0.2,20)
    mValues = np.zeros(alphas.size)
    iters = 1000
    nRealisations = 10
    for k in range(alphas.size):
        a = alphas[k]
        mSum = 0
        for _ in range(nRealisations):
            patterns = common.random_patterns(N,int(a*N))
            W = common.hebbs_rule(patterns)
            ms = np.zeros(iters)
            S = patterns[0]
            for i in range(iters):
                ms[i] = S.dot(patterns[0])/N
                S = nextState(S,W,2)
            mSum += np.average(ms[100:])
        mValues[k] = mSum/nRealisations
    plt.plot(alphas,mValues)
    plt.show()
    #plt.close()

def nextState(S,W,beta=2):
    return np.sign((np.random.rand(S.size) < 1/(1 + np.exp(-2*beta*W.dot(S))))-0.5)
