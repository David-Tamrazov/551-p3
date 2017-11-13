import numpy as np
import pandas as pd
import math
import cv2
from sklearn.model_selection import KFold


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

    def predict(self, X):
        
        # compute the forward pass
        forward_pass_results = self.forward_pass(X)

        # compute the class probabilities based off of the output of the final output layer 
        predicted_probabilities = compute_class_probabilities(forward_pass_results[-1])

        predictions = [] 

        for probabilities in predicted_probabilities:
            predictions.append(np.argmax(probabilities))
        
        return predictions


def preprocess_images(X):
    
    X_preprocessed = []
    
    for idx, image in enumerate(X):
        
        # set the type to uint8 
        img = image.astype(np.uint8)
        
        # remove the noise from the image
        clean_img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
        
        # convert to binary 
        _, bin_img = cv2.threshold(clean_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        np.reshape(bin_img, (4096,))
        
        X_preprocessed.append(bin_img)

        
    X_preprocessed = np.array(X_preprocessed)
    X_preprocessed = np.reshape(X_preprocessed, (1000, 4096))
    return X_preprocessed

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
        X -= np.mean(X, axis=0)

        # normalize the data
        X /= np.std(X, axis=0)

        return X
    
    if (development):
        # Testing code
        df_x = pd.read_csv("train_x.csv", nrows=20000)
        df_y = pd.read_csv("train_y.csv", nrows=20000)
        df_x_t = pd.read_csv("test_x.csv", nrows=20000)

        X_train= df_x.values
        X_test = df_x_t.values
        Y_train = df_y.values
    
    else:
        # Production code - uncomment for submission
        x = np.loadtxt("train_x.csv", delimiter=",")
        y = np.loadtxt("train_y.csv", delimiter=",")
        x_t = np.loadtxt("test_x.csv", delimiter=",")

        X_train = x
        X_test = x_t
        Y_train = y


    # # preprocess the data
    # print "Preprocessing the data..."

    # X_train = preprocess_images(X)
    # X_test = preprocess_images(X_test)

    return X_train, Y_train, X_test


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


def crossvalidation_configurations(stepsizes, epoch_sizes, hidden_layers):
    
    configurations = [] 

    for stepsize in stepsizes:
        for epoch in epoch_sizes:
            for hidden_layer in hidden_layers:
                config = (stepsize, epoch, hidden_layer)
                configurations.append(config)
                
    return configurations


def find_best_parameters(hyperparameter_settings, training_data, training_labels):
    
    setting_performance = []

    # use cross validation to find the optimal hyperparameter setting 
    for setting in hyperparameter_settings:

        # unpack the hyperparameter setting tuple 
        stepsize, epoch, hidden_layers = setting

        # initialize empty lists to score cross validation scores 
        validation_accuracy = [] 
        training_accuracy = []

        # create indices of training data and validation data for k-fold validation
        k_folds = KFold(5, shuffle=True, random_state=69).split(training_labels)

        # create the network
        network = Network(4096, hidden_layers, 40)

        # iterate over k-fold indices 
        for train_idx, val_idx in k_folds:

            for x in range(0, epoch):
                network.train(training_data[train_idx], training_labels[train_idx], stepsize, 0.5)
            
            # get predictions for measuring validation and training accuracy respectively 
            validation_prediction = network.predict(training_data[val_idx])
            training_prediction = network.predict(training_data[train_idx])

            # measure and store the validation & training accuracy  
            validation_accuracy.append((training_labels[val_idx] == validation_prediction).mean())
            training_accuracy.append((training_labels[train_idx] == training_prediction).mean())

        # average the training and validation accuracies over all folds
        avg_train_acc = np.mean(training_accuracy)
        avg_val_acc = np.mean(validation_accuracy)

        print("Stepsize: {} | Epoch Size: {} | Nu Layers: {} | Layer Sizes: {} | Training Accuracy: {} | Validation Accuracy: {}".format(stepsize, epoch, len(hidden_layers), hidden_layers, avg_train_acc, avg_val_acc))
        setting_performance.append((setting, avg_val_acc))

    # get the hyperparameter setting that produced the highest validation accuracy 
    best_setting = max(setting_performance, key=lambda item:item[1])[0]

    return best_setting

def main():

    print "Fetching data..."

    # Production code - uncomment for submission 
    X_train, Y_train, X_test = fetch_data(development=False)

    # # Fetch the data 
    # X_train, Y_train, X_test = fetch_data()

    # reshape the labels into an N x K matrix where N = nu samples and K = nu classes 
    Y_train = create_label_list(Y_train)

    # set the network architecture
    HIDDEN_LAYER_SIZES = [[10], [20, 30], [30, 40, 50]]
    EPOCHS = [10, 20, 30]
    STEPSIZES = [.0001, .0002, .0003]

    configs = crossvalidation_configurations(STEPSIZES, EPOCHS, HIDDEN_LAYER_SIZES)

    print "Running crossvalidation..."

    find_best_parameters(configs, X_train, Y_train)

    print "Finished crossvalidation."

main()
    