# Based off of Keras example. Editted to fit purpose of our assignment
# ==============================================================================

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import preprocessing
from keras.models import model_from_json
from keras.constraints import maxnorm
import numpy as np
import pandas as pd
from numpy import genfromtxt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import train_test_split
import csv

batch_size = 100
num_classes = 40
epochs = 50

# input image dimensions
img_rows, img_cols = 64, 64

print("loading the data set...")
# the data, shuffled and split between train and test sets
# x_train = np.loadtxt('data/train_x.csv', delimiter=',')
# y_train = np.loadtxt('labels.csv', delimiter=',')
# numpy seems to be too slow to load data

# x_train = np.fromfile('data/train_x.csv', dtype=float, sep=',')
# y_train = np.fromfile('labels.csv', dtype=float, sep=',')

# x_train = genfromtxt('data/train_x.csv', delimiter=',')
# y_train = genfromtxt('labels.csv', delimiter=',')

x_train_p = pd.read_csv('data/train_x.csv',sep=',',header=None)
y_train_p = pd.read_csv('labels.csv',sep=',',header=None)
x_train = x_train_p.as_matrix()
y_train = y_train_p.as_matrix()


#####################################################################
#Convert labels to one hot encoding

# classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
# label_arr = []

# # add the label to the 1D array
# for idx, label in enumerate(y_train):
#     label_arr.append(int(classes.index(y_train[idx])))

# y_train = np.eye(40)[label_arr]

# y_train_oneHot = []
# y_train_oh= np.eye(40)[label_arr].tolist()
# for x in y_train_oh:
#     string = ''
#     for i in x:
#         string += str(int(i))
#     y_train_oneHot.append(string)

#####################################################################



print("splitting the training set...")
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")



test_p = pd.read_csv('data/test_x.csv',sep=',',header=None)
test = test_p.as_matrix()

if K.image_data_format() == 'channels_first':
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)
    output_shape = (1, img_rows, img_cols)
else:
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    output_shape = (img_rows, img_cols, 1)

test = test.astype('float32')
test /= 255
print('test shape:', test.shape)
print(test.shape[0], 'test samples')

predictions = model.predict(test)

prediction = pd.DataFrame(predictions)
prediction = prediction.idxmax(axis=1)
print (prediction)


classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

for index, rows in prediction.iteritems():
    prediction[index] = classes[rows]

prediction.index = prediction.index + 1
prediction.to_csv('prediction_1000_samples.csv',index=True)
