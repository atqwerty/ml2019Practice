import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def hypothesis(X, thetas):
    return 1 / (1 + np.exp(-X.dot(thetas)))

def gradient(thetas, X, Y):
    return hypothesis(X, thetas) - Y


dataRaw = np.loadtxt(fname="ex2data1.txt", delimiter=",", usecols=[0, 1, 2])
Y = [dataRaw[i][2] for i in range(len(dataRaw))]
Y = np.asarray(Y, dtype=np.float)
X = np.delete(dataRaw, 2, 1)
# X = X.transpose()

thetas = np.zeros(3, dtype=np.float32)

ax = pl.figure().gca(projection="3d")

# ax.scatter(X[0], X[1], Y)

# X = X.transpose()
X = np.array([np.insert(i, 0, 1) for i in X])

alpha = 0.01
epochs = 1500

for i in range(epochs):
    next_thetas = np.zeros(3).astype(float)
    for j in range(3):
        next_thetas[j] = thetas[j] - (alpha / len(X)) * np.sum((hypothesis(X, thetas) - Y) * np.array(X[:, j]))
    thetas = next_thetas

print(thetas)
print('-----')
# print(hypothesis(X, thetas))
propbs = np.dot(X, thetas)
print(np.sum(propbs) / len(propbs))

# point = np.array([0.0, 0.0, thetas[0]])
# normal = np.array(cross([1, 0, thetas[2]], [0, 1, thetas[1]]))
# xx, yy = np.meshgrid([0, 1], [0, 1])
# z = (-normal[0] * xx - normal[1] * yy + point.dot(normal)) * 1. / normal[2]
# ax.plot_surface(xx, yy, z, alpha=0.2, color="red")

# pl.show()