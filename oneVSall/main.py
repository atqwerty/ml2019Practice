import numpy as np
import matplotlib.pyplot as pl
import csv
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def recieve_data(file):
    with open(file) as csv_file:
        rawData = list(csv.reader(csv_file, delimiter=','))
        data = np.delete(rawData, 0, axis=0)

        return data

data = recieve_data('glass.csv')
np.random.shuffle(data)

Y = [data[i][9] for i in range(len(data))]

X = np.delete(data, 9, axis=1)
X = np.asarray(X, dtype=np.float)

x = X

X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])

# X = np.insert(X, 0, 1, axis=1)
thetas = np.random.rand(6, X.shape[1])
print(np.unique(Y))

alpha = 0.01
epochs = 1500

answer = []
counter = 0

for i in np.unique(Y):
    # thetas = np.random.rand(6, X.shape[1])
    y_copy = []
    for j in range(len(Y)):
        y_copy.append(1 if Y[j] == i else 0)

    for j in range(len(X)):
        output = X.dot(thetas[counter])

        errors = sigmoid(output) - y_copy
        thetas[counter] -= (alpha / len(X)) * errors.dot(X)
    # print(thetas[counter])
    answer.append((thetas[counter], i))
    counter += 1

# print(X.shape)

print(answer)

# train_x = X[:10]

# for i in range(train_x.shape[0]):
#     for j in range(6):
#         print(np.sum(sigmoid(np.dot(train_x[i], answer[j][0].T))) / len(sigmoid(np.dot(train_x[i], answer[j][0].T))))
#         print('-----')
#     print('---------------------------------------')

# print(x[-1])