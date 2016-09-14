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
as possible. The classification error C v for the validation set is defined as
Cv=1 2 p p X μ =1 |ζ(μ) − sgn(O(μ))| where |x| stands for the
absolute value of x , p denotes the number of patterns in the validation set,
and O ( μ ) is the network output for pattern μ. The classification error for
the training set is defined analogously.

a) Normalise the input data to zero mean and unit variance (together for both
the validation and the training set). Train the network without hidden layers.
Use asynchronous updating with learning rate η = 0 . 01, activation function
g(b) = tanh( βb ) with β = 1 / 2.  Initialise the weights randomly with uniform
distribution in [ − 0 . 2 , 0 . 2]. Initialise the biases randomly with uniform
distribution in [ − 1 , 1]. Make 100 independent training experiments, each with
2 · 10 5 iterations (one iteration corresponds to feeding a randomly chosen
pattern and updating weights and thresholds). For each t raining ex- periment
determine the minimum classification error for the training a nd the
classification sets. Average these errors over the independent t raining
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

def run():
    train_data = common.read_data('train_data_2016.txt')
    valid_data = common.read_data('valid_data_2016.txt')

    trainX, trainY = train_data['point']['x'],train_data['point']['y']
    trainL = train_data['sign']

    validX, validY = valid_data['point']['x'],valid_data['point']['y']
    validL = valid_data['sign']

    nTP = len(train_data)
    nVP = len(train_data)
    nPoints = nTP + nVP
    meanX = np.mean(np.append(trainX,validX))
    print(np.mean(np.append(trainX,validX)))
    trainX -= meanX
    validX -= meanX
    print(np.mean(np.append(trainX,validX)))
    print(np.var(np.append(trainX,validX)))


    plt.plot(train_data['point'][train_data['sign'] == 1],'r.')
    plt.plot(train_data['point'][train_data['sign'] == -1],'b.')
