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
def run(iters=100, debug=False):
    digits = common.digits()
    W = common.hebbs_rule(digits)
    P, N = digits.shape

    x_axis = [q/100 for q in range(100)]
    error_mean = []
    error_digit = []
    for q in x_axis:
        error_d = np.zeros(P)
        for n in range(iters):
            distorteds = digits * rp(N, P, q)
            for i, (digit, distorted) in enumerate(zip(digits, distorteds)):
                lucky = True
                updates = 1
                while lucky and not np.array_equal(digit, distorted):
                    temp_test = np.copy(distorted)
                    distorted = astep(W, distorted)
                    lucky = updates % 5 and not np.array_equal(temp_test, distorted)
                    if debug and lucky and not updates % 3:
                        print("Wrong updates were made, still feeling lucky for"
                                " digit ", i, ". Try #", updates, " now...")
                    updates += 1
                error_d[i] += np.sum(np.not_equal(digit, distorted))
                if debug:
                    if not lucky and error_d[i]:
                        err = error_d[i] / ((n+1) * N)
                        print("Was not lucky, no progress after ", updates, " tries"
                                ". Stuck on error ", err, " for digit ", i)
                    if not error_d[i]:
                        print("perfect run! q=", q, " digit ", i, " after ", updates)
        error_digit.append(1 - error_d / (iters * N))
        error_mean.append(np.sum(error_digit[-1]) / P)
    plt.plot(x_axis, error_digit)
    plt.plot(x_axis, error_mean)
    plt.legend(["0", "1", "2", "3", "4", "mean"])
    plt.ylabel("P(pattern | q)")
    plt.xlabel("q")
#     y = expit(np.linspace(-6,6,100))
#     plt.plot(x_axis, y)
    plt.savefig("ex2")
#     plt.show()

def astep(W, digit):
    ilist = [i for i in range(len(W[0]))]
    np.random.shuffle(ilist)
#     for digit in digits:
    for i in ilist:
        digit[i] = np.sign(W[i].T.dot(digit))
    return digit
