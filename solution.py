import os
import sys
import math
import numpy as np

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
    signs = [data['sign'] for data in patterns]
    for i in range(N):
        for j in range(N):
            if i != j:
                prodsum = 0
                for pi in range(p):
                    prodsum += signs[pi][i] * signs[pi][j]
                W[i, j] = wij * prodsum
    print(W[0:3])
    print(np.vdot(signs[0],signs[1]))
    for pi in range(p):
        print(np.sum(np.inner(W,signs[pi])))
    print(np.vdot(signs[0],signs[0]))

def random_patterns(data, N, p):
    patterns = []
    for i in range(p):
        patterns.append(np.random.permutation(data)[:N])
    return patterns

if __name__ == '__main__':
    import sys
    data = read_data('train_data_2016.txt')

    N = [100, 200]
    p = [10, 20, 30, 40, 50, 75, 100, 150, 200]

    patterns = random_patterns(data, N[0], p[8])
    hebbs_rule(patterns, N[0], p[8])
