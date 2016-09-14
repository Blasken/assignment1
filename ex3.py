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
    Ns = [50, 100, 250, 500]
    iters = 400
    nRealisations = 50
    plt.clf()
    for N in Ns:
        ps = np.array([int(i) for i in np.linspace(1,N/2,N/2)])
        print(ps.size)
        alphas = ps/N
        mValues = np.zeros(alphas.size)
        for k in range(alphas.size):
            p = ps[k]
            mSum = 0
            for _ in range(nRealisations):
                patterns = common.random_patterns(N,p)
                W = common.hebbs_rule(patterns)
                ms = np.zeros(iters)
                S = patterns[0]
                for i in range(iters):
                    ms[i] = S.dot(patterns[0])/N
                    S = nextState(S,W,2)
                mSum += np.average(ms[100:])
            mValues[k] = mSum/nRealisations
        plt.plot(alphas,mValues,label="N = {}".format(N))
    plt.ylabel(r'Order parameter, $m_1$',fontsize = 16)
    plt.legend()
    plt.xlabel(r'$\alpha = p/N$',fontsize = 16)
    plt.savefig('ex3a')

def nextState(S,W,beta=2):
    return np.sign((np.random.rand(S.size) < 1/(1 + np.exp(-2*beta*W.dot(S))))-0.5)
