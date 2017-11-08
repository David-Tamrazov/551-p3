import numpy as np
import pandas as pd
import math


class Layer(object):
    
    def __init__(self, input_dim, layer_size):
        self.weights = np.random.randn(input_dim, layer_size) / np.sqrt(input_dim)
        self.bias = np.ones((1, layer_size))
    
    def forward(self, ipt):
        activation = np.dot(ipt, self.weights) + self.bias
        return activation

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

    def backpropagate(self, input_gradient, forward_pass_results):

        backpass_results = []
        parent_gradient = input_gradient

        for idx in reversed( range(0,len(self.layers)) ):

            l = self.layers[idx]
            l_input = forward_pass_results[idx]

            # calculate the weight and bias gradient for this layer
            weight_gradient = np.dot(l_input.T, parent_gradient)
            bias_gradient = np.sum(parent_gradient, axis=0, keepdims=True)
            
            # append the gradient to backpass results for gradient updates later 
            backpass_results = [(weight_gradient, bias_gradient)] + backpass_results

            # set the parent gradient for the next layer in the backprop
            parent_gradient = relu(np.dot(parent_gradient,weight_gradient.T),deriv=True)


        return backpass_results


    def update_weights(self, backprop_results, stepsize):
        
        for idx, result in enumerate(backprop_results):
            
            layer = self.layers[idx]

            weight_gradient = result[0]
            bias_gradient = result[1]

            layer.weights -= stepsize * weight_gradient
            layer.bias -= stepsize * bias_gradient


    def train(self, X_train, Y_train, stepsize, reg_strength):

        # compute the forward pass
        forward_pass_results = self.forward_pass(X_train)

        # compute the class probabilities based off of the output of the final output layer 
        predicted_probabilities = compute_class_probabilities(forward_pass_results[-1])

        # compute the loss
        loss = self.compute_loss(Y_train, predicted_probabilities, reg_strength)

        # compute the initial gradient
        gradient = compute_gradient(Y_train, predicted_probabilities)

        # backpropogate and store the result for weight + bias updates 
        backprop_results = self.backpropagate(gradient, forward_pass_results)

        # update the weights and the bias
        self.update_weights(backprop_results, stepsize)

        return loss

    
    def compute_loss(self, labels, predictions, reg_strength):
        
        # data loss function - keep it generic, allows us to switch out cross entropy for something else
        def compute_data_loss(labels, predictions):
        
            # cross entropy data loss
            num_data = predictions.shape[0]

            ce_loss = -np.log(predictions[range(num_data), labels])

            # return average loss over all data
            return np.sum(ce_loss) / num_data

       
        # regularization loss - keep it generic, allows us to switch out l2 loss for something else
        def compute_reg_loss(reg_strength):
            reg_loss = 0
            
            # add regularization error for each set of weights
            for layer in self.layers:
                reg_loss += 0.5 * reg_strength * np.sum(layer.weights * layer.weights) 
           
            return reg_loss

        # return the total loss combining data and regularization 
        return compute_data_loss(labels, predictions) 


# function to extract labels from lists into a one dimensional array 
def create_label_list(labels):
   
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
  
    # create 1D numpy array of type int 
    label_arr = np.zeros(len(labels), dtype='uint8')

    # add the label to the 1D array
    for idx, label in enumerate(labels):
        label_arr[idx] = classes.index(label[0])

    # return the new 1D array of labels
    return label_arr


def compute_class_probabilities(forward_pass_scores):
    
    exp_scores = np.exp(forward_pass_scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probabilities
    
def fetch_data(development=True):
    
    def standardize(X):
        
        # subtract the mean image and center the data
        X =- X - np.mean(X, axis=0)

        # normalize the data
        X /= np.std(X, axis=0)

        return X
    
    if (development):
        # Testing code
        df_x = pd.read_csv("train_x.csv", nrows=100)
        df_y = pd.read_csv("train_y.csv", nrows=100)

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


def compute_gradient(labels, predictions):
    
    # get the number of examples
    num_data = predictions.shape[0]

    gradient_score = predictions
    gradient_score[range(num_data),labels] -= 1
    gradient_score /= num_data

    return gradient_score


def main():

    print "Fetching data..."

    # # Production code - uncomment for submission 
    # X_train, Y_train = fetch_data(development=False)
    # X_test = np.loadtxt("test_x.csv", delimiter=",")

    # Fetch the data 
    X_train, Y_train = fetch_data()

    # reshape the labels into an N x K matrix where N = nu samples and K = nu classes 
    Y_train = create_label_list(Y_train)

    # set the network architecture
    HIDDEN_LAYER_SIZES = [100]

    # set the termination
    EPOCH_SIZE = 10000

    # set the stepsize
    STEPSIZE = .0001

    # set the regularization strength
    LAMBDA = 0.5

    # create the network
    network = Network(4096, HIDDEN_LAYER_SIZES, 40)

    print "Running training..."

    prev_loss = 100

    for x in range(0, EPOCH_SIZE):
        
        # return the loss for this iteration of training 
        loss = network.train(X_train, Y_train, STEPSIZE, LAMBDA)

        # reduce the learning rate if the loss plateaus
        if prev_loss - loss < 0.01:
            STEPSIZE /= 2
        
        prev_loss = loss
        
        print "Iteration: {} | Loss: {}".format(x, loss)

main()
    