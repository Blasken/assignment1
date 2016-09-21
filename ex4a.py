import os
import sys
import common
import numpy as np
import matplotlib.pyplot as plt

"""
Backpropagation. The task is to write a computer program that solves a
classification task by backpropagation. Train the network on the training
set ‘train data 2016’, and evaluate its performance using the validation set
1 LN stands for the lecture notes. 1 ‘valid data 2016’ (course home page).
Each row corresponds to one patter n. The first two entries in a row give
ξ^μ_1 and ξ^μ_2, the third entry gives the target value for this pattern,
ζ ( μ ) . The aim is is to achieve a classification error that is as small
as possible. The classification error Cv for the validation set is defined as
Cv=1 2 p p X μ =1 |ζ(μ) − sgn(O(μ))| where |x| stands for the
absolute value of x , p denotes the number of patterns in the validation set,
and O ( μ ) is the network output for pattern μ. The classification error for
the training set is defined analogously.

a) Normalise the input data to zero mean and unit variance (together for both
the validation and the training set). Train the network without hidden layers.
Use asynchronous updating with learning rate η = 0.01, activation function
g(b)=tanh(βb) with β=1/2.  Initialise the weights randomly with uniform
distribution in [−0.2, 0.2]. Initialise the biases randomly with uniform
distribution in [−1, 1]. Make 100 independent training experiments, each with
2 · 10^5 iterations (one iteration corresponds to feeding a randomly chosen
pattern and updating weights and thresholds). For each t raining ex- periment
determine the minimum classification error for the training a nd the
classification sets. Average these errors over the independent training
experiments. Discuss your results. ( 1p ).

b) Now use back propagation to train a network with one hidden layer th at has
2 neurons, same activation function and parameters as in a . Perform 100
independent training experiments with up to 2 · 10 5 iterations. For each
experiment find the minimum classification errors for the training and
validation sets.  Determine the average of the minimum errors for both sets.
Repeat for networks with 4, 8, 16, and 32 neurons in the hidden laye r. Plot
the average errors as a function of the number of hidden neurons . Compare
to the results obtained in a . Discuss. Does the hidden layer improve the
performance? What is the effect of increasing the number of hidden neurons?
Do you observe overfitting?
"""

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
        CTerror = np.zeros(iters)
        CVerror = np.zeros(iters)
        for i,p in enumerate(order):
            neurons = runNetwork(trainInput[p],W,theta,actFunc)
            error = trainL[p] - neurons[-1]
            trainNetwork(error,neurons, W, theta, actFuncPrim, learningRate)
            outputT = runNetwork(trainInput,W,theta,actFunc)[-1]
            outputV = runNetwork(validInput,W,theta,actFunc)[-1]

            CTerror[i] = np.not_equal(np.sign(outputT).T,trainL).sum()/len(trainL)
            CVerror[i] = np.not_equal(np.sign(outputV).T,validL).sum()/len(validL)
        averageCT += np.min(CTerror)/nRealisations
        averageCV += np.min(CVerror)/nRealisations

    print("Average minimal training error for {} ".format(averageCT))
    print("Average minimal validation error for {} ".format(averageCV))
    """
    Plotting

    n = 30
    movingAverageCV = np.array([CVerror[i:i+n].sum()/n for i in range(len(CVerror)-n)])
    movingAverageCT = np.array([CTerror[i:i+n].sum()/n for i in range(len(CTerror)-n)])
    plt.clf()
    plt.plot(movingAverageCV,'r-',label='Validation error')
    plt.plot(movingAverageCT,'b-',label='Training error')
    plt.xlabel("Iteration")
    plt.ylabel("Classification error")
    plt.legend()
    plt.savefig("ex4a")

    plt.clf()
    mask = train_data['sign'] == 1
    plt.plot(trainX[mask],trainY[mask],'r.')
    mask = train_data['sign'] == -1
    plt.plot(trainX[mask],trainY[mask],'b.')
    x = np.linspace(-2,2,4)
    y = x*-W[0][0][0]/W[0][0][1] + theta[0][0]
    plt.plot(x,y)
    y = x*W[0][1][0]/W[0][1][1] + theta[0][1]
    plt.plot(x,y)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    """


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
