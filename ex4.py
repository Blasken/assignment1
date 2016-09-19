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

def trainNetwork(error, neurons, weights, biases, actFunc, actFuncPrim, learningRate = 0.01):
    nL = len(weights)
    deltas = list(range(nL))
    deltas[-1] = actFuncPrim(neurons[-1])*error
    for l in range(nL-1):
        currL = l - nL
        deltas[currL] = actFuncPrim(neurons[currL]) * weights[currL+1].dot(deltas[currL+1])
    for l, d in enumerate(deltas):
        weights[l] += learningRate*np.outer(actFunc(neurons[l]),d)
        biases[l] -= learningRate*d

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
    layers = [2,2,1]
    W, theta = initialiseWeights(layers)
    def actFunc(x):
        return np.tanh(x/2)
    def actFuncPrim(x):
        return (1-actFunc(x)**2)/2
    """
    Training
    """
    learningRate = 0.01
    iters = 2*10**5
    order = np.random.randint(len(trainX),size=iters)
    CTerror = np.zeros(iters)
    CVerror = np.zeros(iters)
    print(W)
    for i,p in enumerate(order):
        output, neurons = runNetwork(trainInput[p],W,theta,actFunc)
        error = trainL[p] - output
        trainNetwork(error, neurons, W, theta, actFunc, actFuncPrim, learningRate)
        outputT, _ = runNetwork(trainInput,W,theta,actFunc)
        outputV, _ = runNetwork(validInput,W,theta,actFunc)

        CTerror[i] = np.not_equal(np.sign(outputT).T,trainL).sum()/len(trainL)
        CVerror[i] = np.not_equal(np.sign(outputV).T,validL).sum()/len(validL)
    """
    Plotting
    """
    print(W)
    print(CTerror[-1])
    print(CVerror[-1])
    """
    plt.clf()
    plt.plot(CVerror,'r-',label='Validation error')
    plt.plot(CTerror,'b-',label='Training error')
    plt.legend()
    """
    plt.clf()
    mask = train_data['sign'] == 1
    plt.plot(trainX[mask],trainY[mask],'r.')
    mask = train_data['sign'] == -1
    plt.plot(trainX[mask],trainY[mask],'b.')
    x = np.linspace(-2,2,4)
    print(W[0])
    y = x*W[0][0][1]/W[0][1][1]+ theta[0]
    plt.plot(x,y)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    #"""


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
    currNeurons = inValues
    for i in range(len(weights)):
        neurons[i] = currNeurons
        w, b = weights[i], biases[i]
        currNeurons = currNeurons.dot(w)-b
        actNeuron = actFunction(currNeurons)
    neurons[-1] = currNeurons
    return actNeuron, neurons

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
