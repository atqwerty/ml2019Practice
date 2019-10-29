import numpy as np

class NeuralNetwork:
    def init(self, x, y):
        np.random.seed(1)
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 9)
        self.weights2 = np.random.rand(9, 6)
        self.y = y
        self.output = np.zeros_like(y)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1.0 - z)

    def feed_forward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        print(self.layer1.shape)
        print(self.weights2.shape)
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def back_propogation(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot((self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2