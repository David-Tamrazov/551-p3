import numpy as np
import pandas as pd
import math


class Layer(object):
    
    def __init__(self, input_dim, layer_size):
        self.weights = np.random.randn(input_dim, layer_size) / np.sqrt(input_dim)
        self.bias = np.zeros((1, layer_size))
    
    def forward(self, ipt):
        activation = np.dot(ipt, self.weights) + self.bias
        return activation
 
    def backward(self, gradient):
        self.weights = np.dot(self.weights,gradient)
        self.bias = np.sum(gradent,axis=0, keepdims=True)
        return self.weights

class Network(object):
    
    def __init__(self, data_dim, hidden_layer_sizes, nu_classes):

        # initialize the first layer 
        layers = [ Layer(data_dim, hidden_layer_sizes[0]) ]

        # iterate over the layer sizes
        for x in range(1, len(hidden_layer_sizes)):
            
            # input dim of layer k is the size of layer k-1
            input_dim = hidden_layer_sizes[x-1]

            # size of layer k
            output_dim = hidden_layer_sizes[x]

            # create the layer and append it to the list
            layers.append( Layer(input_dim, output_dim) )
            

        # create the output layer whose output dimension is the number of classes in the data set 
        l_n = Layer(hidden_layer_sizes[-1], nu_classes)
        layers.append(l_n)

        # set the layers of object of the network
        self.layers = layers

    def forward_pass(self, X):
        
        # set pass result of the first layer (input layer) to be the input X- this is what the first hidden layer will take as input 
        pass_results = [X]

        # perform the forward pass with an activation function for all hidden layers 
        for idx in range(0, len(self.layers) - 1):
            
            l = self.layers[idx]
            previous_result = pass_results[idx]

            # feed the previous layer's output as this layer's input
            activation = relu(l.forward(previous_result))

            # store the activation in the pass results list for easier backpropogation later
            pass_results.append(activation)

        
        # output of the final hidden layer 
        o = pass_results[-1]

        final_l = self.layers[-1]

        # final output - pass through the synapse without applying an activation function
        pass_results.append(final_l.forward(o))

        # return the scores
        return pass_results

    def back_pass(self, input_gradient, forward_pass_results):

        backpass_results = len(forward_pass_results)
        parent_gradient = input_gradient

        for idx in reversed(range(0,len(self.layers))):

            l = self.layers[idx]
            l_input = forward_pass_results[idx-1]

            # calculate the weight and bias gradient for this layer
            weight_gradient = np.dot(l_input, parent_gradient)
            bias_gradient = np.sum(parent_gradient, axis=0, keepdims=True)

            # append the gradient to backpass results for gradient updates later 
            backpass_results[idx] = (weight_gradient, bias_gradient)

            # set the parent gradient for the next layer in the backprop
            parent_gradient = relu(np.dot(parent_gradient,weight_gradient.T),deriv=True)


        return backpass_results

    def compute_regularization_loss(self, reg_strength):
        reg_loss += (0.5 * reg_strength * np.sum(layer.weights * layer.weights) for layer in self.layers)
        return reg_loss


def create_label_matrix(labels, nu_classes):
    
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
    
    nu_labels = len(labels)
    
    label_matrix = np.empty((nu_labels, 40))

    for x in range(0, nu_labels):
        
        # get the label
        label = labels[x][0]

        # initialize an array of 0s of size nu_classes
        arr = np.zeros((nu_classes))

        # get the index of the label in the "class order" of labels
        label_idx = classes.index(label)
        # set the entry in the array of 0's that corresponds to that index to 1
        arr[label_idx] = 1
        # add the array to the label matrix
        label_matrix[x] = arr
        
    return label_matrix


def compute_class_probabilities(forward_pass_scores):
    
    exp_scores = np.exp(forward_pass_scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities
    
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
    X_train = standardize(X)

    return X_train, Y_train

def standardize(X):
    
    # subtract the mean image and center the data
    X =- X - np.mean(X, axis=0)

    # normalize the data
    X /= np.std(X, axis=0)

    return X


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

    
def compute_loss(labels, predictions, reg_loss):
    
    def compute_cross_entropy_loss(labels, predictions):
        
        nu_data = len(predictions)

        ce_loss = -np.log(predictions[range(nu_data, y)])

        return np.sum(ce_loss) / nu_data

    avg_ce_loss = compute_cross_entropy_loss(labels, predictions)

    return avg_ce_loss + reg_loss



def compute_gradient(labels, predictions):

    gradient_score = labels - predictions

    return gradient_score

        

def main():

    print "Fetching data..."

    # # Production code - uncomment for submission 
    # X_train, Y_train = fetch_data(development=False)
    # X_test = np.loadtxt("test_x.csv", delimiter=",")

    # Fetch the data 
    X_train, Y_train = fetch_data()

    # reshape the labels into an N x K matrix where N = nu samples and K = nu classes 
    Y_matrix = create_label_matrix(Y_train, 40)

    hidden_layer_sizes = [10, 20, 30]

    network = Network(4096, hidden_layer_sizes, 40)

    pass_results = network.forward_pass(X_train)
    # compute the class probabilities based off of the output of the final output layer 
    class_probabilities = compute_class_probabilities(pass_results[-1])

    gradient = compute_gradient(Y_matrix, class_probabilities)

    network.back_pass(gradient)


main()
    