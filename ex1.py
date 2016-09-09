import os
import sys
import common
import numpy as np

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

def run():
    Ns = [100, 200]
    P = [10, 20, 30, 40, 50, 75, 100, 150, 200]

    for N in Ns:
        for p in P:
            patterns = common.random_patterns(N, p)
            W = common.hebbs_rule(patterns)
# fix algebra....
            print(step(W,patterns))

def step(W, S):
# fix algebra....
    return np.sign(np.dot(W,S))

def p_error(S, S_next):
    pass

if __name__ == '__main__':
#     import sys
#     data = read_data('train_data_2016.txt')

    N = [100, 200]
    p = [10, 20, 30, 40, 50, 75, 100, 150, 200]

#     patterns = random_patterns(data, N[0], p[8])
#     hebbs_rule(patterns, N[0], p[8])
