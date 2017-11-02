import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_hyperparameter_settings():
    reg_strengths = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    tol_ranges = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    hp_settings = []
    for strength in reg_strengths:
        for rng in tol_ranges:
            hp_settings.append((strength, rng))
    
    return hp_settings
    

def find_best_parameters(hyperparameter_settings, training_data, training_labels):

    setting_performance = []

    # use cross validation to find the optimal hyperparameter setting 
    for setting in hyperparameter_settings:

        # unpack the hyperparameter setting tuple 
        reg_strength, tol_range = setting

         # initialize empty lists to score cross validation scores 
        validation_accuracy = [] 
        training_accuracy = []

        # create indices of training data and validation data for k-fold validation
        k_folds = KFold(5, shuffle=True, random_state=69).split(training_labels)

        # iterate over k-fold indices 
        for train_idx, val_idx in k_folds:

            # training; penalty is l2 by default 
            clf = LogisticRegression(C=reg_strength, solver='sag', multi_class='multinomial', max_iter=1000, tol=tol_range, n_jobs=-1)
            clf.fit(training_data[train_idx], training_labels[train_idx].ravel())

            # get predictions for measuring validation and training accuracy respectively 
            validation_prediction = clf.predict(training_data[val_idx])
            training_prediction = clf.predict(training_data[train_idx])

            # measure and store the validation & training accuracy  
            validation_accuracy.append((training_labels[val_idx] == validation_prediction).mean())
            training_accuracy.append((training_labels[train_idx] == training_prediction).mean())

        # average the training and validation accuracies over all folds
        avg_train_acc = np.mean(training_accuracy)
        avg_val_acc = np.mean(validation_accuracy)

        print("Regularization Strength: {} | Tolerance Range: {} | Training Accuracy: {} | Validation Accuracy: {}".format(setting[0], setting[1], avg_train_acc, avg_val_acc))
        setting_performance.append((setting, avg_val_acc))

    # get the regularization strength that produced the highest validation accuracy 
    best_setting = max(setting_performance, key=lambda item:item[1])[0]

    return best_setting


def write_to_file(predictions):
    
    with open("logreg_predictions.csv", 'w') as f:
    
        for idx, prediction in enumerate(predictions):
            line = '{},{}\n'.format(idx, prediction)
            f.write(line)
    
        f.close()


def main():
    
    print  "Loading training and test data..."  

    # # Production code - uncomment for submission
    # x = np.loadtxt("train_x.csv", delimiter=",")
    # y = np.loadtxt("train_y.csv", delimiter=",")
    # x_t = np.loadtxt("test_x.csv", delimiter=",")

    # X = x
    # X_test = x_t
    # Y_train = y

    df_x = pd.read_csv("train_x.csv", nrows=10)
    df_y = pd.read_csv("train_y.csv", nrows=10)
    df_xt = pd.read_csv("test_x.csv", nrows=10)

    X = df_x.values
    X_test = df_xt.values
    Y_train = df_y.values

    print "Preprocessing the data..."
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    print "Finding the best hyperparameter setting through cross validation..."

    # use cross validation to find the optimal hyperparameter setting
    hyperparameter_settings = build_hyperparameter_settings()

    best_c, best_tol = find_best_parameters(hyperparameter_settings, X_train, Y_train)

    print "Finished cross validation for all hyper parameters."

    print "Optimal hyper parameter setting: Regularization Strength: {}, Tolerance Range: {}".format(best_c, best_tol)

    print "Training with optimal hyperparameters..."

    # create indices of training data and validation data for k-fold validation
    k_folds = KFold(5, shuffle=True, random_state=69).split(Y_train)

    training_accuracy = []
    validation_accuracy = []

    # iterate over k-fold indices 
    for train_idx, val_idx in k_folds:
        
        # train classifier with optimal hyperparameters 
        best_clf = LogisticRegression(C=best_c, solver='saga', tol=best_tol, n_jobs=-1)
        best_clf.fit(X_train[train_idx], Y_train[train_idx].ravel())


    print "Creating predictions for test data..."

    predictions = best_clf.predict(X_test)

    print "Writing predictions to file..."

    write_to_file(predictions)

    print "Done."


main()


     




