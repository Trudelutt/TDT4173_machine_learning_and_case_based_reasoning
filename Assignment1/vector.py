import numpy as np
import matplotlib.pyplot as plt
import csv


def createCsv(filepath):
    data = np.genfromtxt(filepath,delimiter=',')
    X = np.zeros((data.shape[0],data.shape[1]))
    X[:,0] = 1
    for i in range (1,data.shape[1]):
        X[:,i] = data[:,i-1]
    return X, data[:,-1]

def create_x_padd(X, dim):
    pad_X = np.zeros((X.shape[0], dim), dtype= float)
    for i in range(X.shape[1]):
        pad_X[:,i] = X[:,i]
    pad_X[:,-1] = X[:,2]* X[:,2]
    pad_X[:,-2] = X[:,1]* X[:,1]
    return pad_X


def create_pos(csv,dim,pos=True):
    ones = 0
    for row in csv[1]:
        if row == 1:
            ones += 1
    zeros = csv[1].shape[0]- ones
    if pos:
        X = np.zeros((ones, dim), dtype= float)
    else:
        X = np.zeros((zeros, dim), dtype= float)
    count = 0
    for i in range(csv[1].shape[0]):
        if(pos and csv[1][i] == 1):
            X[count] = csv[0][i]
            count += 1
        elif not pos and csv[1][i] == 0:
            X[count] = csv[0][i]
            count += 1
    return X
