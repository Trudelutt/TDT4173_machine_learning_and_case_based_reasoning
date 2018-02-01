import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from vector import createX, createY, createCsv, create_pos, create_x_padd

def train(filepath, filepath_test,pad=False):
    csv = createCsv(filepath)
    csv_test = createCsv(filepath_test)
    if(pad):
        X = create_x_padd(csv, 5)
        w = np.array([0.1,0.1,0.1,0.1,0.1])
    else:
        X = createX(csv, 3)
        w = np.array([0.1,0.1,0.1])
    print(X.shape)
    Y = createY(csv,1)
    X_test = createX(csv_test, 3)
    Y_test = createY(csv_test,1)

    pos_x_test, pos_y_test = create_pos(csv, 3,pos=True)
    plt.plot(pos_x_test[:,1], pos_x_test[:,2], "k.")
    neg_x_test, neg_y_test = create_pos(csv, 3, pos=False)
    plt.plot(neg_x_test[:,1], neg_x_test[:,2], "g.")

    pos_x, pos_y = create_pos(csv, 3,pos=True)
    plt.plot(pos_x[:,1], pos_x[:,2], "r.")
    neg_x, neg_y = create_pos(csv, 3, pos=False)
    plt.plot(neg_x[:,1], neg_x[:,2], "b.")

    error_plot = np.zeros((1000,2))
    error_plot_test = np.zeros((1000,2))
    for i in range(1000):
        w = calc_weight(X,Y, 0.1, w)
        error_plot[i] = [i+1, calc_error(w, X,Y)]
        #error_plot_test[i] = [i+1, calc_error(w, X_test, Y_test)]
    #plt.plot(X[:,1], H, 'k')
    print("Weight: ", w)
    return error_plot, error_plot_test,w, X


def calc_weight(X,Y, n, w):
    new_weight = 0
    for i in range(len(X)):
        new_weight += (sigma(w, X[i])-Y[i])*X[i]
    new_weight = w-n*new_weight
    return new_weight

def sigma(W, X):
    z = np.dot(np.transpose(W), X)
    return 1/(1+ np.exp(-z))


def calc_error(w, X, Y):
    e = 0
    for i in range(len(X)):
        e += Y[i]*np.log(sigma(w,X[i])) + (1-Y[i])*np.log(1-sigma(w,X[i]))
    return -1* e/len(X)

def h(X, W):
    H = np. zeros((2,2))
    for i in range(2):
        H[i] = [i, -(W[0]+W[1]*i)/W[2]]
    return H

def circle_h(X, W):
    xs = np.linspace(0.0,1.0,100)
    X1,X2 = np.meshgrid(xs,xs)
    H = np.array(W[0]+ W[1]*X1+ W[2]*X2 + W[3]*X1**2 + W[4]*X2**2 )
    plt.contour(X1,X2, H, [0])



def task2(linear=True):
    filepath = "./datasets/classification/cl_train_2.csv"
    filepath_test = "./datasets/classification/cl_test_2.csv"
    plt.figure()

    if(linear):
        e, e_test, w, X = train(filepath,filepath_test)
        H = h(X, w)
        plt.plot(H[:,0], H[:,1], 'k')
    else:
        e, e_test, w, X = train(filepath,filepath_test,True)
        circle_h(X,w)
    plt.figure()
    plt.plot(e[:,0], e[:,1], 'k')
    plt.show()

def task1():
    filepath = "./datasets/classification/cl_train_1.csv"
    filepath_test = "./datasets/classification/cl_test_1.csv"
    plt.figure()
    e, e_test, w, X = train(filepath,filepath_test)
    H = h(X, w)
    plt.plot(H[:,0], H[:,1], 'k')
    plt.figure()
    print(e[:,[1]])
    print(h(X, w))
    plt.plot(e[:,0], e[:,1], 'k')
    plt.plot(e_test[:,0], e_test[:,1], 'r')
    plt.show()

task2(False)
