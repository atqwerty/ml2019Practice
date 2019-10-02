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

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def hypothesis(X, thetas):
    return 1 / (1 + np.exp(-X.dot(thetas)))

def gradient(thetas, X, Y):
    return hypothesis(X, thetas) - Y


dataRaw = np.loadtxt(fname="ex2data2.txt", delimiter=",", usecols=[0, 1, 2])
Y = [dataRaw[i][2] for i in range(len(dataRaw))]
Y = np.asarray(Y, dtype=np.float)
X = np.delete(dataRaw, 2, 1)
# X = X.transpose()

thetas = np.zeros(28, dtype=np.float32)

# ax = pl.figure().gca(projection="3d")

# ax.scatter(X[0], X[1], Y)

# X = X.transpose()
X = np.array([np.insert(i, 0, 1) for i in X])

X = mapFeature(X[:,0], X[:,1])

# print(X.shape)

alpha = 0.01
epochs = 1500

for i in range(epochs):
    next_thetas = np.zeros(28).astype(float)
    for j in range(28):
        next_thetas[j] = thetas[j] - (alpha / len(X)) * np.sum((hypothesis(X, thetas) - Y) * np.array(X[:, j]))
    thetas = next_thetas
 
print(thetas)

pred = [hypothesis(X, thetas) >= 0.5]
print(np.mean(pred == Y.flatten()) * 100)
