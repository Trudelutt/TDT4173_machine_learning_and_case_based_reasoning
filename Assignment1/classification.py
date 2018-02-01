import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from vector import createCsv, create_pos, create_x_padd

def train(X,Y, X_test,Y_test,pad=False):
    if pad:
        w = np.array([0.1,0.1,0.1,0.1,0.1])
    else:
        w = np.array([0.1,0.1,0.1])
    error_plot = np.zeros((1000,2))
    error_plot_test = np.zeros((1000,2))
    for i in range(1000):
        w = calc_weight(X,Y, 0.1, w)
        error_plot[i] = [i+1, calc_error(w, X,Y)]
        error_plot_test[i] = [i+1, calc_error(w, X_test, Y_test)]
    print("Weight: ", w)
    return error_plot, error_plot_test,w

def get_values_from_file(filepath,pad=False):
    csv = createCsv(filepath)
    if pad:
        X = create_x_padd(csv[0], 5)
    else:
        X = csv[0]
    Y = csv[1]
    pos_x = create_pos(csv, 3,pos=True)
    neg_x = create_pos(csv, 3, pos=False)
    return X,Y,pos_x,neg_x

def calc_weight(X,Y, n, w):
    new_weight = 0
    for i in range(X.shape[0]):
        new_weight += (sigma(w, X[i])-Y[i])*X[i]
    new_weight = w-n*new_weight
    return new_weight

def sigma(W, X):
    z = np.dot(np.transpose(W), X)
    return 1/(1+ np.exp(-z))

def calc_error(w, X, Y):
    e = 0
    for i in range(X.shape[0]):
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

def plot_error(e, e_test):
    plt.figure()
    plt.title = "Error"
    plt.plot(e[:,0], e[:,1], 'k')
    plt.plot(e_test[:,0], e_test[:,1], 'r')

def plot_points(X_pos, X_neg, X, W, title, linear=True):
    plt.figure()
    plt.title = title
    plt.xlabel = "X1"
    plt.ylabel = "X2"
    plt.plot(X_pos[:,1], X_pos[:,2], 'r.')
    plt.plot(X_neg[:,1], X_neg[:,2], 'b.')
    if linear:
        H = h(X, W)
        plt.plot(H[:,0], H[:,1], 'k')
    else:
        circle_h(X,W)

def task2(linear=True):
    filepath = "./datasets/classification/cl_train_2.csv"
    filepath_test = "./datasets/classification/cl_test_2.csv"
    X, Y,x_pos,x_neg = get_values_from_file(filepath, pad = not linear)
    X_test, Y_test,x_pos_test,x_neg_test = get_values_from_file(filepath_test,pad = not linear)
    if(linear):
        e, e_test, w = train(X,Y,X_test,Y_test)
        H = h(X, w)
    else:
        e, e_test, w = train(X,Y,X_test,Y_test,True)
    plot_error(e,e_test)
    plot_points(x_pos, x_neg, X, w, "training result",linear)
    plot_points(x_pos_test, x_neg_test, X_test, w, "test result",linear)
    plt.show()

def task1():
    filepath = "./datasets/classification/cl_train_1.csv"
    filepath_test = "./datasets/classification/cl_test_1.csv"
    X, Y,x_pos,x_neg = get_values_from_file(filepath)
    X_test, Y_test,x_pos_test,x_neg_test = get_values_from_file(filepath_test)
    e, e_test, w = train(X, Y,X_test,Y_test)
    H = h(X, w)
    plot_points(x_pos, x_neg, X, w, "training result")
    plot_points(x_pos_test, x_neg_test, X_test, w, "test result")
    plot_error(e, e_test)
    plt.show()

task2(False)
