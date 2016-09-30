import common
import numpy as np
import matplotlib.pyplot as plt

def trainNetwork(error, neurons, weights, biases, actFuncPrim, learningRate = 0.01):
    """
    Function for updating weights and biases with backpropagation.
    """
    nL = len(weights)
    deltas = list(range(nL))
    deltas[-1] = actFuncPrim(neurons[-1])*error
    for l in -np.arange(nL-1)-2:
        deltas[l] = actFuncPrim(neurons[l]) * weights[l+1].dot(deltas[l+1])
    for l, d in enumerate(deltas):
        weights[l] = weights[l] + learningRate*np.outer(neurons[l],d)
        biases[l] = biases[l] - learningRate*d

def run():
    train_data = common.read_data('train_data_2016.txt')
    valid_data = common.read_data('valid_data_2016.txt')

    trainX, trainY = train_data['point']['x'],train_data['point']['y']
    trainL = train_data['sign'] #Labels

    validX, validY = valid_data['point']['x'],valid_data['point']['y']
    validL = valid_data['sign'] #Labels

    """
    Normalisation
    """

    meanX = np.mean(np.append(trainX,validX))
    trainX -= meanX
    validX -= meanX

    varX = np.var(np.append(trainX,validX))
    trainX /= np.sqrt(varX)
    validX /= np.sqrt(varX)

    meanY = np.mean(np.append(trainY,validY))
    trainY -= meanY
    validY -= meanY

    varY = np.var(np.append(trainY,validY))
    trainY /= np.sqrt(varY)
    validY /= np.sqrt(varY)

    trainInput = np.array([trainX,trainY]).T
    validInput = np.array([validX,validY]).T

    """
    Network
    """
    learningRate = 0.01
    iters = 2 * 10**5
    nRealisations = 100
    averageCT = 0
    averageCV = 0
    def actFunc(x):
        return np.tanh(x/2)
    def actFuncPrim(x):
        return (1-x**2)/2
    layers = [2,1]
    CTerror = np.zeros(iters//100)
    CVerror = np.zeros(iters//100)
    print("Running {} realisations of networks with {} structure...".format(nRealisations,layers))
    for k in range(nRealisations):
        W, theta = initialiseWeights(layers)
        """
        Training
        """

        order = np.random.randint(len(trainX),size=iters)
        CTerror = np.zeros(iters//100)
        CVerror = np.zeros(iters//100)
        for i,p in enumerate(order):
            neurons = runNetwork(trainInput[p],W,theta,actFunc)
            error = trainL[p] - neurons[-1]
            trainNetwork(error,neurons, W, theta, actFuncPrim, learningRate)
            if i%100:
                outputT = runNetwork(trainInput,W,theta,actFunc)[-1]
                outputV = runNetwork(validInput,W,theta,actFunc)[-1]

                CTerror[i//100] = np.not_equal(np.sign(outputT).T,trainL).sum()/len(trainL)
                CVerror[i//100] = np.not_equal(np.sign(outputV).T,validL).sum()/len(validL)
        averageCT += np.min(CTerror)/nRealisations
        averageCV += np.min(CVerror)/nRealisations

    print("Average minimal training error for {} ".format(averageCT))
    print("Average minimal validation error for {} ".format(averageCV))

def runNetwork(inValues,weights,biases,actFunction):
    """
    Run a general feed forward fully connected neural network.
        inValues is the value for the input neurons
        weights and biases are lists of matrices and vectors
        actFuntion is the choosen activation function.

        returns the value of the output neuron, and the preactivated values
                for all neurons.
    """
    neurons = list(range(len(weights)+1))
    neurons[0] = inValues
    for i in range(len(weights)):
        neurons[i+1] = actFunction(neurons[i].dot(weights[i])-biases[i])
    return neurons

def initialiseWeights(layers):
    """
    Creates randomly initialised weight matrices and bias vectors for a
    feed forward network, that is len(layers) deep.

    Takes layers
        ex. [2,1] for a network with 2 inputs and 1 output
            [2,3,1] for a similiar network with one hidden layer with 3 neurons
    Returns weights, biases
    """
    W = [np.random.rand(layers[i],layers[i+1])*0.4 - 0.2 for i in range(len(layers)-1)]
    theta = [np.random.rand(layers[i+1])*2 - 1 for i in range(len(layers)-1)]
    return W, theta
