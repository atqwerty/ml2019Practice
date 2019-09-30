import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import csv

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def hypothesis(X, thetas):
    return 1 / (1 + np.exp(-X.dot(thetas)))

def gradient(thetas, X, Y):
    return hypothesis(X, thetas) - Y

def recieve_data(file):
    with open(file) as csv_file:
        rawData = list(csv.reader(csv_file, delimiter=','))
        data = np.delete(rawData, 0, axis=0)

        return data

data = recieve_data('dataset.csv')

Y = [data[i][3] for i in range(len(data))]
Y = np.asarray(Y, dtype=np.float)
X = np.delete(data, 3, axis=1)
X = np.asarray(X, dtype=np.float)

X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])

thetas = np.zeros(4, dtype=np.float32)
 
# ax = pl.figure().gca(projection="3d")
# ax.scatter(X[0], X[1], X[2], Y)
# pl.show()

X = np.array([np.insert(i, 0, 1) for i in X])

alpha = 0.01
epochs = 1500

for i in range(epochs):
    next_thetas = np.zeros(4).astype(float)
    for j in range(4):
        next_thetas[j] = thetas[j] - (alpha / len(X)) * np.sum((hypothesis(X, thetas) - Y) * np.array(X[:, j]))
    thetas = next_thetas

print(thetas)
# print('-----')
# # print(hypothesis(X, thetas))
probs = np.dot(X, thetas)
print(probs)
print(np.sum(probs) / len(probs))

train_data_raw = recieve_data('train.csv')

train_x = np.asarray(train_data_raw, dtype=np.float)

train_x = np.array([[(train_x[i][j]-np.mean(train_x[:, j]))/(np.max(train_x[:, j])-np.min(train_x[:, j])) for j in range(len(train_x[i]))] for i in range(len(train_x))])

train_x = np.array([np.insert(i, 0, 1) for i in train_x])

print(np.dot(train_x, thetas))

# # point = np.array([0.0, 0.0, thetas[0]])
# # normal = np.array(cross([1, 0, thetas[2]], [0, 1, thetas[1]]))
# # xx, yy = np.meshgrid([0, 1], [0, 1])
# # z = (-normal[0] * xx - normal[1] * yy + point.dot(normal)) * 1. / normal[2]
# # ax.plot_surface(xx, yy, z, alpha=0.2, color="red")

# # pl.show()