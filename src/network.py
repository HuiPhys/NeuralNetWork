import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.layer_num = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in list(zip(sizes[:-1], sizes[1:]))]

    def feedforward(self, a):
        for w, b in list(zip(self.weights, self.biases)):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n_training = len(training_data)
        for count_epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n_training, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(count_epoch, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(count_epoch))
    
    def update_mini_batch(self, mini_batch, eta):
        n_mini_batch = len(mini_batch)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            cost_to_weights, cost_to_biases = self.backprop(x, y)
            nabla_w = [nw + cw for nw, cw in list(zip(nabla_w, cost_to_weights))]
            nabla_b = [nb + cb for nb, cb in list(zip(nabla_b, cost_to_biases))]
        self.weights = [w - (eta/n_mini_batch)* nw for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [b - (eta/n_mini_batch)* nb for b, nb in list(zip(self.biases, nabla_b))]

    def backprop(self, x, y):
        activation = x
        activations = [x]
        zs = []
        cost_to_biases = [np.zeros(b.shape) for b in self.biases]
        cost_to_weights = [np.zeros(w.shape) for w in self.weights]
        for w, b in list(zip(self.weights, self.biases)):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
        delta_l = np.multiply(self.cost_derivative(activations[-1], y), sigmoid_prime(zs[-1]))
        cost_to_biases[-1] = delta_l
        cost_to_weights[-1] = np.outer(delta_l, activations[-2])
        for count in range(2, self.layer_num):
            w = self.weights[-count+1]
            delta_l = np.multiply(np.dot(np.transpose(w), delta_l), sigmoid_prime(zs[-count]))
            cost_to_biases[-count] = delta_l
            cost_to_weights[-count] = np.outer(delta_l, activations[-count-1])
        return (cost_to_weights, cost_to_biases)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return (1.0 - sigmoid(z))*sigmoid(z)