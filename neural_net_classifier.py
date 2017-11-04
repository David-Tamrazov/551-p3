import numpy as np
import pandas as pd


class Neuron(object):
    def __init__ (self, bias, initial_weight_val):
        self.bias = np.full((64,64), bias)
        self.weights = np.full((64, 64), initial_weight_val)

    def output(self, ipt):
        activation = np.dot(ipt, self.weights.T) + self.bias
        return relu(activation)

class Layer(object):
    def __init__(self, neurons):
        self.neurons = neurons
    
    def layer_output(self, ipt): 
        return [ neuron.output(ipt) for neuron in self.neurons ]


def fetch_data(development=True):
    
    if (development):
        # Testing code
        df_x = pd.read_csv("train_x.csv", nrows=1000)
        df_y = pd.read_csv("train_y.csv", nrows=1000)

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

    preprocessed_data = np.empty(training_data.shape)

    for idx, img in enumerate(training_data):
        
        # add the preprocessed image to the preprocessed_data array
        preprocessed_data[idx] = (img - np.mean(img)) / np.std(img)
    
    # print "Training Data: {} |\n Preprocessed data: {}".format(training_data, preprocess_data)
    return preprocessed_data

def sig(x, deriv=False):
    
    if (deriv):
        return x*(1-x)

    return 1 / (1 + np.exp(-x))

def relu(x, deriv=False):
    
    if (deriv):
        dx = np.zeros(x.shape)
        dx[(x >= 0)] = 1
        return dx

    return np.maximum(x, 0)


def main():

    print "Fetching data..."

    # # Production code - uncomment for submission 
    # X_train, Y_train = fetch_data(development=False)
    # X_test = np.loadtxt("test_x.csv", delimiter=",")

    # Fetch the data 
    X_train, Y_train = fetch_data()

    # Create a neuron 
    n = Neuron(0.1, 5.)

    print "Running through the neurons..."


    alpha = 1.0

    for x in xrange(1,10):

        for idx, img in enumerate(X_train):
            
            est = n.output(img)

            err = (Y_train[idx] - est)
            grad = relu(est, True)

            update = - (err) * grad * img
            # print "Est: {}".format(est)
            # print "Err: {} \n\n Grad: {} \n\n".format(err, grad)
            # print "Update: {}".format(update)
            # print "Weights pre update: {}".format(n.weights)
            n.weights += alpha * update
            # print "Weights post update: {}".format(n.weights)



main()
    