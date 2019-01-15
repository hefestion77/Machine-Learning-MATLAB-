# In IDLE, if we hadn't downloaded the required modules from PIP:
     # import sys
     # print(sys.executable)
     # In Terminal or cmd go to the printed directory and write 
     # "python3.7 -m pip install --user numpy pandas matplotlib scipy sklearn"
# ----------------------------------------------------------------------------------------------------------------------------------------------




# NEURAL NETWORK ALGORITHM WITH PYTHON

# We must import modules numpy, pandas, matplotlib, scipy and scikit-learn:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

def featurenormalise(X, y = None):
    """ (np.ndarray of floats) -> np.ndarray of floats

    Returns a ndarray that is a mean normalised and scaled version of X
    """
    mu = np.mean(X, axis = 0) # 0 means for every column (feature)
    std = np.std(X, axis = 0) # standard deviations for every column (feature)

    return (X - mu) / std

def randInitialWeights(L_in, L_out):
    """ (int, int) -> np.array of floats
    
    Returns an array L_out x (L_in + 1) of random initial weights where L_in
    is the number of input units and L_out the number of output units 
    in the layers adjacent to the matrix of weights
    
    """
    epsilon = np.sqrt(6) / np.sqrt((L_in + L_out))
    return np.random.rand(L_out, 1 + L_in) * (2 * epsilon) - epsilon

def xavier_he_init(L_in, L_out):
    """ (int, int) -> np.array of floats
    
    Returns an array (L_out x (L_in + 1)) of random initial weights using
    Xavier/He initialisation method, where L_in
    is the number of input units and L_out the number of output units 
    in the layers adjacent to the matrix of weights
    
    """
    xav_std = np.sqrt(np.sqrt(2/(L_in + L_out)))
    return np.random.normal(0, xav_std, size = (L_out, 1 + L_in))
    
def sigmoid(z):
    """ (np.array of floats) -> np.array of floats
    
    Returns an array computed from the sigmoid activation function of z
    
    """
    
    return (1 + np.exp(-z)) ** (-1)

def sigmoidGrad(z):
    """ (np.array of floats) -> np.array of floats
    
    Returns an array computed from the first partial
    derivate of the sigmoid activation function of z
    
    """
    
    return sigmoid(z) * (1 - sigmoid(z))


def addBias(X):
    """ (np.array of floats) -> np.array of floats
    
    Adds bias (column of ones on the left of array X) to intercept weight0
    
    """
    m = X.shape[0]
    bias = np.ones((m,1))
    
    return np.hstack((bias, X)) # X = np.concatenate((bias, X), axis = 1)



if __name__== '__main__':
    mat_contents = sio.loadmat('ex4data1.mat') # To load arrays from a .mat file
    # sio.whosmat('????.mat') -----> to see the contents of a .mat file
    X = mat_contents['X']
    # X = featurenormalise(X)
    y = mat_contents['y']
    
    y[y==10] = 0 # because indexes in MATLAB start at 1, in Python at 0.
                 # this way it's not necessary to convert number 0 to 10 
                 # to match it with the correspondent index

    m, n = X.shape # n = num of features, m = num of training examples

    num_layers = 3
    num_hidden_units = n
    num_output_units = 10
    
    weights1 = randInitialWeights(n, num_hidden_units)
    weights2 = randInitialWeights(num_hidden_units, num_output_units)
    
    initial_nn_params = np.concatenate((weights1.flatten(), weights2.flatten()))
    
    a0 = addBias(X) # adds bias to intercept weight0
    
    # X = np.delete(X, 0, 1) if we'd want to remove the bias in X 
            # (removes the first column of 1s, index 0, in X)
    
    z1 = a0 @ weights1.T
    a1 = sigmoid(z1)  # initial values of the hidden units
    a1 = addBias(a1)
    
    z2 = a1 @ weights2.T
    h = sigmoid(z2)
    
    labels = np.arange(num_output_units)
    unrolled_y = (y == labels)
    
    delta2 = h - unrolled_y
    delta1 = (delta2 @ weights2) * addBias(sigmoidGrad(z1))
    delta1 = np.delete(delta1, 0, 1)
    
    DELTA2 = delta2.T @ a1
    DELTA1 = delta1.T @ a0
    
    # find the best lambda using CV set
    
    lambd = 0
    

    weights1toreg = weights1[:, 1:]
    weights2toreg = weights2[:, 1:]
    
    regulterm = (lambd/(2*m)) * (np.sum(weights1toreg**2) + np.sum(weights2toreg**2))
    J = (1/m) * (np.sum(-(unrolled_y * np.log(h))-(1-unrolled_y)*np.log(1-h))) + regulterm
    
    weights1grad = DELTA1 / m + (lambd/m)*(np.hstack((np.zeros((weights1.shape[0], 1)), weights1toreg)))
    weights2grad = DELTA2 / m + (lambd/m)*(np.hstack((np.zeros((weights2.shape[0], 1)), weights2toreg)))
    
    grad = np.concatenate((weights1grad.flatten(), weights2grad.flatten()))
