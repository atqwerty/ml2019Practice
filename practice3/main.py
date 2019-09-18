import numpy as np
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D

dataRaw = pandas.read_csv('train_data.csv')
dataRaw = dataRaw.to_numpy()
Y = [dataRaw[i][4] for i in range(len(dataRaw))]
X = np.delete(dataRaw, 4, 1)


thetas = np.zeros(len(dataRaw[0]))
alpha = 0.01
epochs = 1800

Y = [(i-np.mean(Y))/(np.max(Y)-np.min(Y)) for i in Y]
X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])
X = np.array([np.insert(i, 0, 1) for i in X])

def cost(x, y, thetas):
    h = x.dot(thetas)
    return np.sum(np.square(h - y)) / (2 * len(x))

costs = []
dataLength = len(X)

for i in range(epochs):
    next_thetas = np.zeros(5).astype(float)
    for j in range(5):
        next_thetas[j] = thetas[j] - (alpha / dataLength) * \
        np.sum((X.dot(thetas) - Y) * np.array(X[:, j]))
    thetas = next_thetas

    costs.append(cost(X, Y, thetas))

print(costs)

Xrange = range(epochs)

plt.plot(Xrange, costs)

plt.ylabel('Cost')
plt.xlabel('Epochs')

plt.show()