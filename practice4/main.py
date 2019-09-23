import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def hypothesis(X, thetas):
    return 1 / (1 + np.exp(-X.dot(thetas)))

def gradient(thetas, X, Y):
    return np.dot(hypothesis(X, thetas) - Y, X) / Y.shape[0]


dataRaw = np.loadtxt(fname="ex2data1.txt", delimiter=",", usecols=[0, 1, 2])
Y = [dataRaw[i][2] for i in range(len(dataRaw))]
Y = np.asarray(Y, dtype=np.float)
X = np.delete(dataRaw, 2, 1)
X = X.transpose()

thetas = np.zeros(3, dtype=np.float32)

ax = pl.figure().gca(projection="3d")

ax.scatter(X[0], X[1], Y)

# pl.show()

X = X.transpose()
X = np.array([np.insert(i, 0, 1) for i in X])

alpha = 0.01
epochs = 1500

# print(hypothesis(X, thetas))
# print(gradient(thetas, X, Y))

for i in range(epochs):
    next_thetas = np.zeros(3).astype(float)
    for j in range(3):
        next_thetas[j] = thetas[j] - (alpha / len(X)) * gradient(thetas, X, Y)[j]

    thetas = next_thetas

print(thetas)

# print(hypothesis(X, thetas))