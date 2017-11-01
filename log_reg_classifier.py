import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# x = np.loadtxt("train_x_test.csv", delimiter=",")
# y = np.loadtxt("train_y_test.csv", delimiter=",")

# Production code - uncomment for submission
x = np.loadtxt("train_x.csv", delimiter=",")
y = np.loadtxt("train_y.csv", delimiter=",")
x_t = np.loadtxt("test_x.csv", delimiter=",")

# reshape the images to 64x64
X_train = x.reshape(-1, 64, 64) 
Y_train = y.reshape(-1, 1)
X_test = x_t.reshape(-1, 64, 64)

reg_strengths = [1 / X_train.shape(0), 10 / X_train.shape(0), 100 / X_train.shape(0), 1000 / X_train.shape(0), 10000 / X_train.shape(0)]
max_iterations = [10, 100, 1000, 10000, 100000]
tol_ranges = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
scaler = StandardScaler()

# preprocess the data by subtracting the mean (centering) and scaling it down to unit size 
scaler.transform(X_train)
scaler.transform(Y_train)

# create indices of training data and validation data for k-fold validation
k_folds = KFold(5, shuffle=True, random_state=69).split(Y_train)

c_performance = []
max_iter_performance = []
tol_performance = []

print("Running cross validation to determine optimal regularization strength...")
# use cross validation to find the optimal regularization strength 
for c in reg_strengths:

    # initialize empty lists to score cross validation scores 
    validation_accuracy = [] 
    training_accuracy = []

    # iterate over k-fold indices 
    for train_idx, val_idx in k_folds:

        # training; penalty is l2 by default 
        clf = LogisticRegression(C=c, solver='saga', max_iter=1000, tol=1e-4, n_jobs=-1)
        clf.fit(X_train[train_idx], Y_train[train_idx])

        # get predictions for measuring validation and training accuracy respectively 
        validation_prediction = clf.predict(X_train[val_idx])
        training_prediction = clf.predict(X_train[train_idx])

        # measure and store the validation & training accuracy  
        validation_accuracy.append((Y_train[val_idx] == validation_prediction).mean())
        training_accuracy.append((Y_train[train_idx] == training_prediction).mean())

    # average the training and validation accuracies over all folds
    avg_train_acc = np.mean(training_accuracy)
    avg_val_acc = np.mean(validation_accuracy)

    print("C: {} | Training Accuracy: {} | Validation Accuracy: {}".format(c, avg_train_acc, avg_val_acc))
    c_performance.append((c, avg_val_acc))

# get the regularization strength that produced the highest validation accuracy 
best_c = max(c_performance, key=lambda item:item[1])[0]

print("Optimal regularization strength: {}".format(best_c))

print("Running cross validation to determine optimal max iterations...")

for itr in max_iterations:
    
     # initialize empty lists to score cross validation scores 
    validation_accuracy = [] 
    training_accuracy = []

    # iterate over k-fold indices 
    for train_idx, val_idx in k_folds:

        # training; penalty is l2 by default 
        clf = LogisticRegression(C=best_c, solver='saga', max_iter=itr, tol=1e-4, n_jobs=-1)
        clf.fit(X_train[train_idx], Y_train[train_idx])

        # get predictions for measuring validation and training accuracy respectively 
        validation_prediction = clf.predict(X_train[val_idx])
        training_prediction = clf.predict(X_train[train_idx])

        # measure and store the validation & training accuracy  
        validation_accuracy.append((Y_train[val_idx] == validation_prediction).mean())
        training_accuracy.append((Y_train[train_idx] == training_prediction).mean())

    # average the training and validation accuracies over all folds
    avg_train_acc = np.mean(training_accuracy)
    avg_val_acc = np.mean(validation_accuracy)

    print("Max Iterations: {} | Training Accuracy: {} | Validation Accuracy: {}".format(itr, avg_train_acc, avg_val_acc))
    max_iter_performance.append((itr, avg_val_acc))

# keep track of the max iterator
best_iter = max(max_iter_performance, key=lambda item:item[1])[0]

print("Optimal number of max iterations: {}".format(best_iter))

print("Running cross validation to determine optimal tolerance range...")

for tol in tol_ranges:
    
     # initialize empty lists to score cross validation scores 
    validation_accuracy = [] 
    training_accuracy = []

    # iterate over k-fold indices 
    for train_idx, val_idx in k_folds:

        # training; penalty is l2 by default 
        clf = LogisticRegression(C=best_c, solver='saga', max_iter=best_iter, tol=tol, n_jobs=-1)
        clf.fit(X_train[train_idx], Y_train[train_idx])

        # get predictions for measuring validation and training accuracy respectively 
        validation_prediction = clf.predict(X_train[val_idx])
        training_prediction = clf.predict(X_train[train_idx])

        # measure and store the validation & training accuracy  
        validation_accuracy.append((Y_train[val_idx] == validation_prediction).mean())
        training_accuracy.append((Y_train[train_idx] == training_prediction).mean())

    # average the training and validation accuracies over all folds
    avg_train_acc = np.mean(training_accuracy)
    avg_val_acc = np.mean(validation_accuracy)

    print("Tolerance Range: {} | Training Accuracy: {} | Validation Accuracy: {}".format(tol, avg_train_acc, avg_val_acc))
    tol_performance.append((tol, avg_val_acc))

# get the regularization strength that produced the highest validation accuracy 
best_tol = max(tol_performance, key=lambda item:item[1])[0]

print("Optimal tolerance range: {}".format(best_tol))

print("Finished cross validation for all hyper parameters.")

print("Creating predictions for test data...")

# Get predictions using our cross-validated hyperparameters 
best_clf = LogisticRegression(C=best_c, solver='saga', max_iter=best_iter, tol=best_tol, n_jobs=-1)
test_pred = clf.predict(X_test)

print("Writing predictions to file...")

# save predictions to file 
write_to_file(test_pred)

print("Done.")


def write_to_file(predictions):
    return 0



    


     




