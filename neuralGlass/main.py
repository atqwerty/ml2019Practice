import numpy as np
import matplotlib.pyplot as plt
import pandas
import nn

data = pandas.read_csv('glass.csv')

data = data.sample(frac=1).reset_index(drop=True)

labels = data['Type']
data = data.drop('Type', axis = 1)

data = data.to_numpy()
labels = labels.to_numpy()

one_hot_labels = np.zeros((214, 9))

for i in range(214):
    one_hot_labels[i, labels[i]] = 1

X = data

X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])

model = nn.NeuralNetwork()
model.init(X, one_hot_labels, 15000)
model.train()