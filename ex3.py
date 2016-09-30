import common
import numpy as np
import matplotlib.pyplot as plt

def run():
    Ns = [50, 100, 250, 500]
    iters = 300
    nRealisations = 100
    plt.clf()
    for N in Ns:
        alphas = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.26, 0.3])
        ps = np.array([int(i) for i in alphas*N])
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
    plt.savefig('ex3b')

def nextState(S,W,beta=2):
    return np.sign((np.random.rand(S.size) < 1/(1 + np.exp(-2*beta*W.dot(S))))-0.5)
