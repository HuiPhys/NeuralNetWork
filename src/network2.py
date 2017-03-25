# import standard libraries
import random
import json
import sys

# import third party library
import numpy as np

#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):
    
    @staticmethod
    def fn(a, y):
        return 0.5*numpy.linalg.norm(a - y )**2

    @staticmethod
    def delta(a, y, z):
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
    
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y)* np.log(1 - a)))

    @staticmethod
    def delta(a, y, z):
        return (a-y)



class Network(object):
    def __init__(self, sizes, cost = CrossEntropyCost):
        self.layer_num = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in list(zip(self.sizes[:-1], self.sizes[1:]))]
   
    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in list(zip(self.sizes[:-1], self.sizes[1:]))]
   
    def feedforward(self, a):
        for w, b in list(zip(self.weights, self.biases)):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda  = 0.0,
            evaluation_data = None,
            monitor_evaluation_cost = False, 
            monitor_evaluation_accuracy= False,
            monitor_training_cost= False,
            monitor_training_accuracy= False):
    
        if evaluation_data: n_evaluation = len(evaluation_data)
        n_training = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for count_epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n_training, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n_training)
            print ("Epoch %s training complete" % count_epoch)   
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {} / {}".format(accuracy, n_evaluation))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(accuracy, n_training))
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        n_mini_batch = len(mini_batch)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            cost_to_weights, cost_to_biases = self.backprop(x, y)
            nabla_w = [nw + cw for nw, cw in list(zip(nabla_w, cost_to_weights))]
            nabla_b = [nb + cb for nb, cb in list(zip(nabla_b, cost_to_biases))]
        self.weights = [w * (1 - eta * lmbda/n) - (eta/n_mini_batch)* nw for w, nw in list(zip(self.weights, nabla_w))]
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
        delta_l = self.cost.delta(activations[-1], y, zs[-1])
        cost_to_biases[-1] = delta_l
        cost_to_weights[-1] = np.outer(delta_l, activations[-2])
        for count in range(2, self.layer_num):
            w = self.weights[-count+1]
            delta_l = np.multiply(np.dot(np.transpose(w), delta_l), sigmoid_prime(zs[-count]))
            cost_to_biases[-count] = delta_l
            cost_to_weights[-count] = np.outer(delta_l, activations[-count-1])
        return (cost_to_weights, cost_to_biases)


    def evaluate(self, data):
        data_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in data_results)
        
        
    def accuracy(self, data, convert = False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper."""
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else: 
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
            
        
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        if convert:
            results = [self.cost.fn(self.feedforward(x), vectorized_result(y)) for (x, y) in data]
        else: 
            results = [self.cost.fn(self.feedforward(x), y) for (x, y) in data]
        weights_sum = np.sum([np.sum(np.square(weight)) for weight in self.weights])
        return (1.0/len(data))*np.sum(results)+(lmbda/(2.0*len(data)))*weights_sum 

    def save_data(self, filename):
        data = {"sizes": self.sizes, 
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        with open(filename, 'w') as fp:
            json.dump(data, fp) 
    
#### Miscellaneous functions
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    res = np.zeros(10)
    res[j] = 1
    return res

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return (1.0 - sigmoid(z))*sigmoid(z)