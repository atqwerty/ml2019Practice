import numpy as np
import matplotlib.pyplot as pl
import csv

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def recieve_data(file):
    with open(file) as csv_file:
        rawData = list(csv.reader(csv_file, delimiter=','))
        data = np.delete(rawData, 0, axis=0)

        return data

data = recieve_data('glass.csv')

Y = [data[i][9] for i in range(len(data))]

X = np.delete(data, 9, axis=1)
X = np.asarray(X, dtype=np.float)

X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])

X = np.insert(X, 0, 1, axis=1)
thetas = np.zeros(X.shape[1], dtype=np.float32)

classifiers = np.zeros(shape=(6, 10))

alpha = 0.01
epochs = 1500

answer = []

for i in np.unique(Y):
    y_copy = np.where(Y == i, 1, 0)

    for _ in range(epochs):
        output = X.dot(thetas)

        errors = sigmoid(output) - y_copy
        thetas -= alpha / len(X) * errors.dot(X)
    answer.append((thetas, i))
print(answer)