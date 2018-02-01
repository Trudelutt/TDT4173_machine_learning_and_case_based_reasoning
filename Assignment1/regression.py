import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from vector import createCsv

#Calculate ordinary least squares to find weights
def OLS(x,y):
    w = np.dot(np.transpose(x), x)
    w = np.linalg.pinv(w)
    w = np.dot(w, np.transpose(x))
    w = np.dot(w,y)
    return w

def plot_result(X,Y,Xt,Yt,H,Ht):
    plt.figure()
    plt.title("training data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X[:,1], Y, "b.")
    plt.plot(X[:,1], H, 'k')
    plt.figure()
    plt.title("Test data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(Xt[:,1], Yt, "r+")
    plt.plot(Xt[:,1], Ht, 'k')


def calc_h(W,X):
    H = np.zeros(X.shape[0], dtype= float)
    for i in range(len(X)):
        for j in range(len(X[i])):
            H[i] += X[i][j]*W[j]
    return H

#Calculate the mean squared error
def calc_error(H,X,Y):
    e = 0
    for i in range(len(Y)):
        e += (H[i]-Y[i])**2
    return e/len(Y)

def train(filepath):
    csv = createCsv(filepath)
    X = csv[0]
    Y = csv[1]
    W= OLS(X,Y)
    H = calc_h(W,X)
    return X, Y, W, H

def test_data(filepath):
    csv = createCsv(filepath)
    X = csv[0]
    Y = csv[1]
    return X,Y

"""
X: X vector for the training dataset
Y: Y vector for the training dataset
Xt: X vector for the test dataset
Yt: Y vector for the test dataset
H: estimated Y value vector
"""
def print_end_info(X,Y,Xt,Yt,H):
    print("Weights ", W)
    print("Error", calc_error(H,X,Y))
    print ("Error of testdata ", calc_error(H, Xt, Yt))


filepath = "./datasets/regression/train_1d_reg_data.csv"
X, Y, W, H = train(filepath)
Xt,Yt = test_data("./datasets/regression/test_1d_reg_data.csv")
Ht = calc_h(W,Xt)
plot_result(X,Y,Xt,Yt,H,Ht)
print_end_info(X,Y,Xt,Yt,H)
plt.show()
