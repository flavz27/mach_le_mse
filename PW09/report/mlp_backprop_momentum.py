#!/usr/bin/python3
import numpy as np

class MLP:
    """
    This code was adapted from:
    https://rolisz.ro/2013/04/18/neural-networks-in-python/
    """
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        return 1.0 - a**2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        return a * ( 1 - a )

    @property
    def n_inputs(self):
        return self.layers[0]

    @property
    def n_outputs(self):
        return self.layers[len(self.layers) - 1]

    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        self.layers = layers
        if activation == 'logistic':
            self.activation = self.__logistic
            self.activation_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.activation = self.__tanh
            self.activation_deriv = self.__tanh_deriv

        self.init_weights()

    def init_weights(self):
        self.weights = []
        for i in range(1, len(self.layers) - 1):
            self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i] + 1)) - 1) * 0.25)
        self.weights.append((2 * np.random.random((self.layers[i] + 1, self.layers[i + 1])) - 1) * 0.25)

        #self.weights = np.array(self.weights)

    def fit(self, train_set, test_set=None, learning_rate=0.1, momentum=0.5, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = train_set[0]
        y = train_set[1]


        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)
        to_return = np.zeros(epochs)
        to_return_test = np.zeros(epochs)

        last_weight = []
        for i in range(len(self.weights)):
            last_weight.append(self.weights[i] * 0)

        for k in range(epochs):
            temp = np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])
                a = [X[i]]

                for l in range(len(self.weights)):
                    a.append(self.activation(np.dot(a[l], self.weights[l])))
                error = y[i] - a[-1]

                temp[it] = np.mean(error ** 2)
                deltas = [error * self.activation_deriv(a[-1])]

                for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                deltas.reverse()
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    last_weight[i] = learning_rate * layer.T.dot(delta) + momentum * last_weight[i]
                    self.weights[i] += last_weight[i]


            to_return[k] = np.mean(temp)
            if test_set:
                predicted = np.zeros(len(test_set[1]))
                for i in np.arange(len(test_set[0])):
                    predicted[i] = (self.predict(test_set[0][i]) - test_set[1][i]) ** 2


                to_return_test[k] = np.mean(predicted)

        if test_set:
            return to_return, to_return_test

        return to_return

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def compute_MSE(self, dataset):
        X = dataset[0]
        y = dataset[1]

        output = np.zeros(y.shape)
        for i in np.arange(X.shape[0]):
            output[i] = self.predict(X[i])

        return (np.mean((y - output) ** 2), output)
