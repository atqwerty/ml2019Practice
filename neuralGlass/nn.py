import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def init(self, X, Y, epochs):
        np.random.seed(42)
        self.X = X
        self.one_hot_encoding = Y
        self.hidden_weights = np.random.randn(9, 9)
        self.output_weights = np.random.randn(9, 6)
        self.epochs = epochs
        self.error_cost = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

    def test_feed_forward(self):
        sigmoid_hypothesis = self.sigmoid(np.dot(self.X, self.hidden_weights))
        softmax_hypothesis = self.softmax(np.dot(sigmoid_hypothesis, self.output_weights))

        return softmax_hypothesis

    def train(self):
        for i in range(self.epochs):
            # Feed forward
            sigmoid_hypothesis = self.sigmoid(np.dot(self.X, self.hidden_weights))
            softmax_hypothesis = self.softmax(np.dot(sigmoid_hypothesis, self.output_weights))

            # Back propogation
            d_output_weights = np.dot(sigmoid_hypothesis.T, softmax_hypothesis - self.one_hot_encoding)
            d_hidden_weights = np.dot(self.X.T, self.sigmoid_derivative(np.dot(self.X, self.hidden_weights)) * np.dot(softmax_hypothesis - self.one_hot_encoding, self.output_weights.T))

            self.hidden_weights -= d_hidden_weights * 0.01
            self.output_weights -= d_output_weights * 0.01

            # if i % 200 == 0:
            loss = np.sum(-self.one_hot_encoding * np.log(softmax_hypothesis))
            print('Loss function value: ', loss)
            self.error_cost.append(loss)
