import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out

def hypothesis(X, thetas):
    return 1 / (1 + np.exp(-X.dot(thetas)))

def gradient(thetas, X, Y):
    return hypothesis(X, thetas) - Y

dataRaw = np.loadtxt(fname="ex2data2.txt", delimiter=",", usecols=[0, 1, 2])
Y = [dataRaw[i][2] for i in range(len(dataRaw))]
Y = np.asarray(Y, dtype=np.float)
X = np.delete(dataRaw, 2, 1)

thetas = np.zeros(28, dtype=np.float32)

X = np.array([np.insert(i, 0, 1) for i in X])
X = mapFeature(X[:,1], X[:,2])

alpha = 0.1
epochs = 100000

for i in range(epochs):
    difference = hypothesis(X, thetas) - Y
    thetas -= (alpha / len(X)) * X.T.dot(difference)
 
print('Thetas: ' + str(thetas))
print('-----')

pred = [hypothesis(X, thetas) >= 0.5]
print('Accruracy is: ' + str(np.mean(pred == Y.flatten()) * 100) + '%')
