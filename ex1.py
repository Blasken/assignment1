import os
import sys
import math
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

def read_data(path):
    f = open(path)
    data = []

    for line in f:
        x, y, state = line.split()
        data.append(((float(x), float(y)), int(state)))

    npdata = np.array(data,dtype=[('point', [('x', np.float),('y', np.float)]),
                                  ('sign', np.int)])
    return npdata

def hebbs_rule(patterns, N, p):
    wij = 1/N
    W = np.zeros((N,N))
#     signs = [data['sign'] for data in patterns]
    for i in range(N):
        for j in range(N):
            if i != j:
                for p_i in range(p):
                    W[i, j] +=patterns[p_i][i] *patterns[p_i][j]
    W[i, j] = wij * W[j, j]
    print(W[0:3])
#     print(np.vdot(signs[0],signs[1]))
#     for pi in range(p):
#         print(np.sum(np.inner(W,signs[pi])))
#     print(np.vdot(signs[0],signs[0]))

def random_patterns(data, N, p):
    patterns = []
    for i in range(p):
        #patterns.append(np.random.permutation(data)[:N])
        patterns.append(np.random.choice([-1,1],N))
    return patterns

if __name__ == '__main__':
    import sys
    data = read_data('train_data_2016.txt')

    N = [100, 200]
    p = [10, 20, 30, 40, 50, 75, 100, 150, 200]

    patterns = random_patterns(data, N[0], p[8])
    hebbs_rule(patterns, N[0], p[8])
