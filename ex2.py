import numpy as np
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
def run(steps=100):
    digits = common.digits()
    W = common.hebbs_rule(digits)

#     for q in range(0.1, 0.9, 0.001):
#         p = random_patterns(len(digits[0]), len(digits), q)

#     distorted = digits * rp(len(digits[0]), len(digits), 0.3)

    distorted_list = [digits * rp(len(digits[0]), len(digits), q)
                                 for q in np.linspace(0.1,0.9,9)]

    ilist = [i for i in range(len(W[0]))]

    for q in np.linspace(0.1, 0.5, 10):
        errors = []
        error = 0
        print("q =", q)
        for n in range(steps):
            distorted = digits * rp(len(digits[0]), len(digits), q)
#             print(np.sum(np.not_equal(distorted, digits)))
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
            error += (np.sum(np.not_equal(digits, distorted))) / steps
        errors.append(error / digits.size)
        print(error / digits.size)
    print(errors)
    print()

def astep(W, digits):
    ilist = [i for i in range(len(W[0]))]
    np.random.shuffle(ilist)
    for digit in digits:
        for i in ilist:
            digit[i] = np.sign(W[i].T.dot(digit))
    return digits
