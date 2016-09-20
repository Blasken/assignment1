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
def run(iters=100):
    digits = common.digits()
    W = common.hebbs_rule(digits)

    x_axis = [q/100 for q in range(100)]
    errors = []
    for q in x_axis:
        error = 0
#         print("q =", q)
        for n in range(iters):
            distorted = digits * rp(len(digits[0]), len(digits), q)
            distorted = astep(W, distorted)
#         np.random.shuffle(ilist)
#         for digit in distorted:
#             for i in ilist:
#                 digit[i] = astep(W[i],digit)
#                 if not i % 40:
#                 plt.imshow(digit.reshape((16,10)))
#                 plt.show()
#             plt.imshow(digit.reshape((16,10)))
#             plt.show()
#             print(np.sum(np.not_equal(digits, distorted)))
            error += np.sum(np.not_equal(digits, distorted))
        errors.append(error / (iters * digits.size))
    plt.plot(x_axis,errors)
    y = expit(np.linspace(-6,6,100))
    plt.plot(x_axis, y)
    plt.show()

def astep(W, digits):
    ilist = [i for i in range(len(W[0]))]
    np.random.shuffle(ilist)
    for digit in digits:
        for i in ilist:
            digit[i] = np.sign(W[i].T.dot(digit))
    return digits
