import numpy as np
import sys
import matplotlib.pyplot as plt

def read_data(path):
    """
    Read in a file from path
    Takes: path
    Returns a numpy array indexed as array[point[x,y],sign]
        example:
        point = array[point] gives [x,y]
        sign = array[sign] gives [+-1]
    """
    f = open(path)
    data = []

    for line in f:
        x, y, state = line.split()
        data.append(((float(x), float(y)), int(state)))

    npdata = np.array(data,dtype=[('point', [('x', np.float),('y', np.float)]),
                                  ('sign', np.int)])
    return npdata


def hebbs_rule(patterns):
    """
    Calculates Wij using Hebb's rule

    Takes patterns
    Returns Wij
    """
#    p = len(patterns)
#    N = len(patterns[0])
#    wij = 1/N
#    W = np.zeros((N,N))
#    for i in range(N):
#        for j in range(N):
#            if i != j:
#                for p_i in range(p):
#                    W[i, j] += patterns[p_i][i] *patterns[p_i][j]
#   return wij * W
    W = patterns.T.dot(patterns)/len(patterns[0])
    return W - np.diag(np.diag(W)) # Removing diagonal

def random_patterns(N, P, q=0.5):
    """
    Takes N = pattern length, p = |patterns|
    Returns numpy array of shape (p,N) with values [-1,1]
    """
#     patterns = []
#     for i in range(p):
        #patterns.append(np.random.permutation(data)[:N])
#         patterns.append(np.random.choice([-1,1],N))
#     [np.random.choice([-1,1],N) for _ in range(p)]
    return np.random.choice([-1,1],N*P, p=[q, 1-q]).reshape(P,N)
#     return patterns

class Digit(np.ndarray):
    def __new__(cls, input_array, number=None):
        obj = np.asarray(input_array).view(cls)
        obj.number = number
        obj.bits = len(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.number = getattr(obj, 'number', None)
        self.bits = getattr(obj, 'bits', None)

    def show(self):
        plt.imshow(self.reshape(16,10))
        plt.show()

    def distort(self, q):
        return self * random_patterns(self.bits, 1, q)

def digits():
    p0 = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1, 1, 1, 1, 1, 1, 1,-1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1, 1, 1, 1,-1,-1, 1, 1, 1,-1,
          -1,-1, 1, 1, 1, 1, 1, 1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

    p1 = [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
          -1,-1,-1, 1, 1, 1, 1,-1,-1,-1]

    p2 = [ 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
           1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
          -1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
          -1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
          -1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
          -1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
          -1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
           1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
           1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
           1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
           1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
           1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
           1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
           1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
           1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
           1, 1, 1, 1, 1, 1, 1, 1,-1,-1]

    p3 = [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,
          -1,-1, 1, 1, 1, 1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1, 1, 1, 1, 1,-1,-1,
          -1,-1,-1,-1, 1, 1, 1, 1,-1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
          -1,-1, 1, 1, 1, 1, 1, 1, 1,-1,
          -1,-1, 1, 1, 1, 1, 1, 1,-1,-1]

    p4 = [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
          -1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
          -1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
          -1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
          -1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
          -1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
          -1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
          -1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
          -1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
          -1,-1,-1,-1,-1,-1,-1, 1, 1,-1]

    npp0 = np.array(p0, dtype=np.int)
    npp1 = np.array(p1, dtype=np.int)
    npp2 = np.array(p2, dtype=np.int)
    npp3 = np.array(p3, dtype=np.int)
    npp4 = np.array(p4, dtype=np.int)
#     npp0 = Digit(p0, number=0, dtype=np.int)
#     npp1 = Digit(p1, number=1, dtype=np.int)
#     npp2 = Digit(p2, number=2, dtype=np.int)
#     npp3 = Digit(p3, number=3, dtype=np.int)
#     npp4 = Digit(p4, number=4, dtype=np.int)

    return np.array([npp0, npp1, npp2, npp3, npp4])
