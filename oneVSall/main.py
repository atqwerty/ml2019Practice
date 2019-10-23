import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from copy import deepcopy
import pandas

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, thetas):
    h = sigmoid(X.dot(thetas))
    cost_1 = np.log(h)
    cost_2 = np.array([np.log(1-i) for i in h])
    summ = y.dot(cost_1) + np.array([1-i for i in y]).dot(cost_2)
    return -summ/len(X)

data = pandas.read_csv('glass.csv')

data = data.sample(frac=1).reset_index(drop=True)

Y = data['Type']
data = data.drop('Type', axis = 1)

data = data.to_numpy()
Y = Y.to_numpy()

# np.random.shuffle(data)

X = data

x = X

X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])

X = np.array([np.insert(i, 0, 1) for i in X])

# np.random.seed(1)

thetas = np.random.randn(6, 10) * 0.001

y = deepcopy(Y)

rate = 0.01
alpha = 0.7
epochs = 1000

answer = []
labels = [1, 2, 3, 5, 6, 7]

costs = np.zeros((6, epochs))

for i in range(6):
    y_copy = deepcopy(Y)
    idxs_i = y_copy == i + 1
    y_copy[idxs_i] = 1
    y_copy[~idxs_i] = 0

    for j in range(epochs):
        output = X.dot(thetas[i])

        errors = sigmoid(output) - y_copy
        holder = X.T.dot(errors) / len(X)
        holder += (np.sum(thetas[i]) - thetas[i][0]) / len(X) * alpha
        thetas[i] -= rate * holder
        costs[i][j] = cost_function(X, y_copy, thetas[i])
    answer.append((thetas[i], i))

for type in range(6):
        plt.plot(range(len(costs[type])), costs[type])
        plt.show()

train_x = X[:10]

for i in range(train_x.shape[0]):
    for j in range(6):
        print(str(sigmoid(np.dot(train_x[i], answer[j][0]))) + " " + str(labels[j]))
        print('-----')
    print('---------------------------------------')

print(x[-1])

# rres = 0.0
# for type in range(6):
#     for i in range(len(Y)):        
#         y[i] = 1 if Y[i] == (type + 1) else 0

#     pred = [sigmoid(np.dot(X, thetas[type])) >= 0.5]
#     print(np.mean(pred == y.flatten()) * 100)
#     rres += np.mean(pred)


