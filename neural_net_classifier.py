import numpy as np
import pandas as pd


class Neuron(object):
    def __init__ (self, bias, intial_weight_val):
        self.bias = bias
        self.weights = np.full((64, 64), intial_weight_val)

    def output(self, ipt):
        activation = np.sum(self.weights.T * ipt) + self.bias
        return sig(activation)

class Layer(object):
    def __init__(self, neurons=[]):
        self.neurons = neurons
    
    def layer_output(self, ipt): 
        return [ neuron.output(ipt) for neuron in self.neurons ]


def fetch_data(development=True):
    
    if (development):
        # Testing code
        df_x = pd.read_csv("train_x.csv", nrows=10)
        df_y = pd.read_csv("train_y.csv", nrows=10)

        X = df_x.values
        Y_train = df_y.values
    
    else:
        # Production code - uncomment for submission
        x = np.loadtxt("train_x.csv", delimiter=",")
        y = np.loadtxt("train_y.csv", delimiter=",")
        x_t = np.loadtxt("test_x.csv", delimiter=",")

        X = x
        X_test = x_t
        Y_train = y


    # preprocess the data
    print "Preprocessing the data..."
    X_train = preprocess_data(X)

    # Reshape the data
    X_train = X_train.reshape(-1, 64, 64) # reshape 
    Y_train = Y_train.reshape(-1, 1) 

    return X_train, Y_train


def preprocess_data(training_data):
    return (training_data - np.mean(training_data)) / np.std(training_data)

def sig(x, deriv=False):
    
    if (deriv):
        return x*(1-x)

    return 1 / (1 + np.exp(-x))

def main():

    print "Fetching data..."

    # # Production code - uncomment for submission 
    # X_train, Y_train = fetch_data(development=False)
    # X_test = np.loadtxt("test_x.csv", delimiter=",")

    # Fetch the data 
    X_train, Y_train = fetch_data()

    # Create a neuron 
    n = Neuron(0.1, 1.)

    print "Running through the neurons..."


    alpha = 2.0
    for x in xrange(1,10000):

        for idx, img in enumerate(X_train):
            
            est = n.output(img)
            err = (Y_train[idx] - est)

            update = - (err) * sig(est, True) * img
            n.weights += alpha * update

            print "Update: {}".format(update)


main()
    