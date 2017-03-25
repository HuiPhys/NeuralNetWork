"""
overfitting
~~~~~~~~~~~

Plot graphs to illustrate the problem of overfitting.  
"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import mnist_loader
import network2

# Third-party libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

def main(filename, num_epochs,
         training_cost_xmin=200, 
         test_accuracy_xmin=200, 
         test_cost_xmin=0, 
         training_accuracy_xmin=0,
         training_set_size=1000, 
         lmbda=0.0):
    """``filename`` is the name of the file where the results will be
    stored.  ``num_epochs`` is the number of epochs to train for.
    ``training_set_size`` is the number of images to train on.
    ``lmbda`` is the regularization parameter.  The other parameters
    set the epochs at which to start plotting on the x axis.
    """
    run_network(filename, num_epochs, training_set_size, lmbda)
    make_plots(filename, num_epochs, 
               test_accuracy_xmin,
               training_cost_xmin,
               test_accuracy_xmin, 
               training_accuracy_xmin,
               training_set_size)

def run_network(filename, num_epochs, training_set_size=1000, lmbda=0.0):
    """Train the network for ``num_epochs`` on ``training_set_size``
    images, and store the results in ``filename``.  Those results can
    later be used by ``make_plots``.  Note that the results are stored
    to disk in large part because it's convenient not to have to
    ``run_network`` each time we want to make a plot (it's slow).
    """
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
    test_cost, test_accuracy, training_cost, training_accuracy = \
         net.SGD(training_data[:training_set_size], num_epochs, 
         	     mini_batch_size=100, eta = 0.5, lmbda = lmbda,
                 evaluation_data = test_data,
                 monitor_evaluation_cost = True, 
                 monitor_evaluation_accuracy= True,
                 monitor_training_cost= True,
                 monitor_training_accuracy= True)
    with open(filename, 'w') as fp:
            json.dump([test_cost, test_accuracy, training_cost, training_accuracy], fp) 
    
def make_plots(filename, num_epochs, 
               training_cost_xmin=200, 
               test_accuracy_xmin=200, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0,
               training_set_size=1000):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)
#    plot_test_cost(test_cost, num_epochs, test_cost_xmin)
#    plot_training_accuracy(training_accuracy, num_epochs, 
#                           training_accuracy_xmin, training_set_size)
#    plot_overlay(test_accuracy, training_accuracy, num_epochs,
#                 min(test_accuracy_xmin, training_accuracy_xmin),
#                 training_set_size)
    
def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    plt.figure(1)
    plt.subplot(111)
    plt.plot(range(training_cost_xmin, num_epochs), training_cost[training_cost_xmin:num_epochs])
    plt.xlabel('epoch number')
    plt.ylabel('training cost')
    plt.grid(True)
    # plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.show()

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    plt.figure(2)
    plt.subplot(211)
    plt.plot(range(test_accuracy_xmin, num_epochs), test_accuracy[test_accuracy_xmin:num_epochs])
    plt.xlabel('epoch number')
    plt.ylabel('training accuracy')
    plt.grid(True)
    # plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.show()

#def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    
    
#def plot_training_accuracy(training_accuracy, num_epochs, 
#                           training_accuracy_xmin, training_set_size):

#def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,
#                 training_set_size):


if __name__ == "__main__":
    filename = input("Enter a file name: ")
    num_epochs = int(input(
        "Enter the number of epochs to run for: "))
    training_cost_xmin = int(input(
        "training_cost_xmin (suggest 200): "))
    test_accuracy_xmin = int(input(
        "test_accuracy_xmin (suggest 200): "))
    test_cost_xmin = int(input(
        "test_cost_xmin (suggest 0): "))
    training_accuracy_xmin = int(input(
        "training_accuracy_xmin (suggest 0): "))
    training_set_size = int(input(
        "Training set size (suggest 1000): "))
    lmbda = float(input(
        "Enter the regularization parameter, lambda (suggest: 5.0): "))
    main(filename, num_epochs, training_cost_xmin, 
         test_accuracy_xmin, test_cost_xmin, training_accuracy_xmin,
         training_set_size, lmbda)