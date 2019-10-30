import numpy as np
import matplotlib.pyplot as plt
import pandas

np.random.seed(3)
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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

instances = X.shape[0]
attributes = X.shape[1]
hidden_nodes = 9
output_labels = 9

hidden_weights = np.random.rand(attributes,hidden_nodes)
bias_hidden = np.random.randn(hidden_nodes)

output_weights = np.random.rand(hidden_nodes,output_labels)
bias_output = np.random.randn(output_labels)
learning_rate = 0.01

error_cost = []

for epoch in range(15000):
    # Feedforward

    # To hidden layer
    z_hidden = np.dot(X, hidden_weights) + bias_hidden
    hypothesis = sigmoid(z_hidden)

    # To output layer
    z_output = np.dot(hypothesis, output_weights) + bias_output
    normalized_probability_distribution = softmax(z_output)

    # Back Propagation

    # Recalculate output weights

    output_cost = normalized_probability_distribution - one_hot_labels
    z_output_weights = hypothesis

    output_cost_weights = np.dot(z_output_weights.T, output_cost)

    output_bias_cost = output_cost

    # Recalculate hidden weights

    z_output_hypothesis = output_weights
    cost_hypothesis = np.dot(output_cost , z_output_hypothesis.T)
    z_hidden_hypothesis = sigmoid_derivative(z_hidden)
    z_hidden_weights = X
    hidden_weigths_cost = np.dot(z_hidden_weights.T, z_hidden_hypothesis * cost_hypothesis)

    hidden_bias_cost = cost_hypothesis * z_hidden_hypothesis

    # Update Weights

    hidden_weights -= learning_rate * hidden_weigths_cost
    bias_hidden -= learning_rate * hidden_bias_cost.sum(axis=0)

    output_weights -= learning_rate * output_cost_weights
    bias_output -= learning_rate * output_bias_cost.sum(axis=0)

    if epoch % 200 == 0:
        loss = np.sum(-one_hot_labels * np.log(normalized_probability_distribution))
        print('Loss function value: ', loss)
        error_cost.append(loss)