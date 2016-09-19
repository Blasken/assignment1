import numpy as np
import common

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
def run():
    digits = common.digits()
    W = common.hebbs_rule(digits)

#     for q in range(0.1, 0.9, 0.001):
#         p = random_patterns(len(digits[0]), len(digits), q)

    distorted = digits * common.random_patterns(len(digits[0]), len(digits), 0.3)
    print(distorted)
    print(np.sum(np.not_equal(distorted, digits)))

    for digit in distorted:
        astep(W,digit)

def astep(W, digit):
    pass
