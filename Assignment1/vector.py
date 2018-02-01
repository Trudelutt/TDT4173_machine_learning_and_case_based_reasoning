import numpy as np
import matplotlib.pyplot as plt
import csv


#mycsv = csv.reader(open(filepath))
#count = 0



def createCsv(filepath):
    f = open(filepath)
    mycsv = csv.reader(f)
    data = []
    for row in mycsv:
        data.append(row)
    f.close()
    return data

def create_x_padd(csv, dim):
    X = np.zeros((len(csv), dim), dtype= float)
    count = 0
    for row in csv:
        x = [1]
        x.extend(row[0:-1])
        x.append(0)
        x.append(0)
        X[count] = np.array(x)
        count += 1
    X[:,-1] = X[:,2]* X[:,2]
    X[:,-2] = X[:,1]* X[:,1]
    print(X)
    return X

def createX(csv, dim):
    X = np.zeros((len(csv), dim), dtype= float)
    count = 0
    for row in csv:
        x = [1]
        x.extend(row[0:-1])
        X[count] = np.array(x)
        count += 1
    return X

def createY(csv, dim):
    Y = np.zeros((len(csv), dim), dtype=float)
    count = 0
    for row in csv:
        Y[count] = np.array(row[-1])
        count += 1
    return Y

def create_pos(csv,dim,pos=True):
    ones = 0
    for row in csv:
        if row[-1] == "1.000":
            ones += 1
    zeros = len(csv)- ones
    if pos:
        X = np.zeros((ones, dim), dtype= float)
        Y = np.zeros((ones, 1), dtype= float)
    else:
        X = np.zeros((zeros, dim), dtype= float)
        Y = np.zeros((zeros, 1), dtype= float)
    count = 0
    for row in csv:
        if(pos and row[-1] == "1.000"):
            Y[count] = np.array(row[-2])
            x = [1]
            x.extend(row[0:-1])
            X[count] = np.array(x)
            count += 1
        elif not pos and row[-1] == "0.000":
            Y[count] = np.array(row[-1])
            x = [1]
            x.extend(row[0:-1])
            X[count] = np.array(x)
            count += 1
    return X,Y

#filepath = "./datasets/classification/cl_train_2.csv"
#csv = createCsv(filepath)
#X = create_x_padd(csv, 5)
#print(X.shape)
#I = createX(csv, 3)
#print(I.shape)
#for row in X:
#    print (row)
#print(createY(csv,1))
