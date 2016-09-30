import numpy as np
from scipy.special import expit
import common
import matplotlib.pyplot as plt
from common import random_patterns as rp

"""
Recognising digits with the deterministic Hopfield model.Thetask is to
recognise distorted versions of the digits shown in Fig. 1 using the
deterministic Hopfield model with asynchronous updating.  Create binary
representations of the digits shown in Fig. 1 and store these patterns.
Then distort the patterns by flipping a fraction q of randomly chosen bits.
Feed these patterns to the network and use asynchronous updating. For each
digit, repeat many times. Determine and plot the probability that the
network retrieves the correct pattern as a function of q. Discuss.
"""
def run(iters=100, show_updates=False):
    digits = common.digits()
    W = common.hebbs_rule(digits)
    P, N = digits.shape

    x_axis = [q/100 for q in range(101)]
    error_mean = []
    error_digit = []
    updates_digit = []
    for q in x_axis:
        error_d = np.zeros(P)
        updates_d = np.copy(error_d)
        for n in range(iters):
            distorteds = digits * rp(N, P, q)
            for i, (digit, distorted) in enumerate(zip(digits, distorteds)):
                lucky = True
                updates = 0
                while lucky and not np.array_equal(digit, distorted):
                    temp_test = np.copy(distorted)
                    distorted = astep(W, distorted)
                    updates += 1
                    lucky = updates % 10 and not np.array_equal(temp_test, distorted)
                error_d[i] += np.sum(np.not_equal(digit, distorted))
                if show_updates:
                    updates_d[i] += updates
        error_digit.append(1 - error_d / (iters * N))
        error_mean.append(np.sum(error_digit[-1]) / P)
        if show_updates:
            updates_digit.append(updates_d / iters)
    fig, errax = plt.subplots()
    errax.plot(x_axis, error_digit)
    errax.plot(x_axis, error_mean)
    errax.legend(["0", "1", "2", "3", "4", "mean"])
    errax.set_ylabel("P(pattern | q)")
    errax.set_xlabel("q")
    plt.savefig("ex2")
    if show_updates:
        updax = errax.twinx()
        updax.plot(x_axis, updates_digit, ls='--')
        updax.legend(["updates 0", "updates 1", "updates 2", "updates 3", "updates 4"], loc=4)
        updax.set_ylabel("Updates")
        plt.savefig("ex2u")
    plt.show()

def astep(W, digit):
    ilist = [i for i in range(len(W[0]))]
    np.random.shuffle(ilist)
    for i in ilist:
        digit[i] = np.sign(W[i].T.dot(digit))
    return digit
