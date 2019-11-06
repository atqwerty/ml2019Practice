import numpy as np

class NeuralNetwork:
    def init(self, X, one_hot_labels, epochs):
        # np.random.seed(42)
        self.X = X
        self.one_hot_labels = one_hot_labels
        self.instances = self.X.shape[0]
        self.attributes = self.X.shape[1]
        self.hidden_nodes = 9
        self.output_labels = 6
        self.hidden_weights = np.random.rand(self.attributes, self.hidden_nodes)
        self.bias_hidden = np.random.randn(self.hidden_nodes)
        self.output_weights = np.random.rand(self.hidden_nodes, self.output_labels)
        self.bias_output = np.random.randn(self.output_labels)
        self.learning_rate = 0.001
        self.error_cost = []
        self.epochs = epochs
        # print(self.hidden_weights)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid (x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

    def train(self):
        for i in range(self.epochs):
            # Feedforward

            # To hidden layer
            z_hidden = np.dot(self.X, self.hidden_weights) + self.bias_hidden
            hypothesis = self.sigmoid(z_hidden)

            # To output layer
            z_output = np.dot(hypothesis, self.output_weights) + self.bias_output
            normalized_probability_distribution = self.softmax(z_output)

            # Back Propagation

            # Recalculate output weights
            output_cost = normalized_probability_distribution - self.one_hot_labels
            z_output_weights = hypothesis

            output_cost_weights = np.dot(z_output_weights.T, output_cost)

            output_bias_cost = output_cost

            # Recalculate hidden weights
            z_output_hypothesis = self.output_weights
            cost_hypothesis = np.dot(output_cost , z_output_hypothesis.T)
            z_hidden_hypothesis = self.sigmoid_derivative(z_hidden)
            z_hidden_weights = self.X
            hidden_weigths_cost = np.dot(z_hidden_weights.T, z_hidden_hypothesis * cost_hypothesis)

            hidden_bias_cost = cost_hypothesis * z_hidden_hypothesis

            # Update Weights
            self.hidden_weights -= self.learning_rate * hidden_weigths_cost
            self.bias_hidden -= self.learning_rate * hidden_bias_cost.sum(axis=0)

            self.output_weights -= self.learning_rate * output_cost_weights
            self.bias_output -= self.learning_rate * output_bias_cost.sum(axis=0)

            # if i % 200 == 0:
            #     loss = np.sum(-self.one_hot_labels * np.log(normalized_probability_distribution))
            #     print('Loss function value: ', loss)
            #     self.error_cost.append(loss)

        print(normalized_probability_distribution)