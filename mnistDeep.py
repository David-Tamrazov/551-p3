# Based off of Tensorflow beginner MNIST tutorial. Editted to fit purpose of our assignment
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import csv
import re
from sklearn import preprocessing
import tempfile
import pandas as pd
import time

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


pathTrainX = "data/train_x.csv"
pathTrainY = "data/train_y.csv"
pathTestX = "data/test_x.csv"

def output_predict_to_file(predict_y,output_filename):
    with open(output_filename,"w") as output:
        output.write("Id,Label")
        output.write("\n")
        print("Predict_y is: ",predict_y)
        for el in range(len(predict_y)):
            output.write(str(el+1))
            output.write(",")
            output.write(str(predict_y[el]))  
            output.write("\n")
    print("output file complete")

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 64, 64, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([10, 10, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([10, 10, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([16 * 16 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 40])
    b_fc2 = bias_variable([40])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
  # Import data

  #Loads data in. Already flattened into [50,000][4096]
  # train_x = np.loadtxt(pathTrainX, delimiter=",")
  # train_y = np.loadtxt(pathTrainY, delimiter=",")
  # test_x = np.loadtxt(pathTestX, delimiter=",")

  train_x_p = pd.read_csv('data/train_x.csv',sep=',',header=None)
  train_y_p = pd.read_csv('data/train_y.csv',sep=',',header=None)
  test_x_p = pd.read_csv('data/test_x.csv',sep=',',header=None)

  train_x = train_x_p.as_matrix()
  train_y = train_y_p.as_matrix()
  test_x = test_x_p.as_matrix()

  train_y_oneHot = []

#####################################################################
  #Convert labels to one hot encoding

  classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
  label_arr = []

  # add the label to the 1D array
  for idx, label in enumerate(train_y):
      label_arr.append(int(classes.index(train_y[idx])))


  train_y_oneHot = np.eye(40)[label_arr]


#####################################################################

#Check to ensure data is in proper format
  print("Train x is: ", train_x)
  print("Length of train x is: ", len(train_x[0]))
  print("Train Y is: ",train_y)
  print("Length of train y is: ",len(train_y_oneHot))
  print("Test x is: ", test_x)
  print("Length of test x is: ", len(test_x[0]))
  print(train_y_oneHot[0])

  smallx = train_x[0:100]
  smally = train_y_oneHot[0:100]

  # Create the model w/ 64x64 = 4096 pixels and 40 classes
  x = tf.placeholder(tf.float32, [None, 4096])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 40])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    accPh = []
    train_accuracy =0
    for j in range(30):
      start_time = time.time()
      for i in range(499):
        batch_xs = train_x[(i*100):((i+1)*100)]
        batch_ys = train_y_oneHot[(i*100):((i+1)*100)]
        #Add accuracy result to the list
        #accPh.append(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}))        
        #train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
      print("Epoch completed: ",j)
      for i in range(499):
        batch_xs = train_x[(i*100):((i+1)*100)]
        batch_ys = train_y_oneHot[(i*100):((i+1)*100)]
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (j, train_accuracy))
      print(time.time()-start_time)

      #if i % 50 == 0:
      print("Step i is: ",i) 

    #print('test accuracy %g' % accuracy.eval(feed_dict={x: train_x[0:2500], y_: train_y[0:2500], keep_prob: 1.0}))
      #Run on test set
    #result = []
    prediction = tf.argmax(y_conv, 1)
    #result = prediction.eval(feed_dict={x: batchTest, keep_prob: 0.0})
    batchTest = test_x[0:2500]
    print("Batch 1 is: ",batchTest)
    batchTest2 = test_x[2500:5000]
    batchTest3 = test_x[5000:7500]
    batchTest4 = test_x[7500:10000]
    #print("Batch 2 is: ",batchTest2)
    #print("Len of batch 1: ",len(batchTest))
    #print("Len of batch 2: ",len(batchTest2))
    l1 = prediction.eval(feed_dict={x: batchTest, keep_prob: 1.0})
    #print("l1 is: ",l1)
    l2 = prediction.eval(feed_dict={x: batchTest2, keep_prob: 1.0})
    l3 = prediction.eval(feed_dict={x: batchTest3, keep_prob: 1.0})
    l4 = prediction.eval(feed_dict={x: batchTest4, keep_prob: 1.0})
    #print("l2 is: ",l2)
    re1 = np.concatenate((l1, l2), 0)
    re2 = np.concatenate((re1, l3), 0)
    result = np.concatenate((re2, l4), 0)    
    #result.append(sess.run(tf.argmax(y_conv, 1), feed_dict={x: batchTest, keep_prob: 0.0}))
    #result.append(sess.run(tf.argmax(y_conv, 1), feed_dict={x: batchTest2, keep_prob: 0.0}))

    #write to file
    output_predict_to_file(result, "cnnresults.csv")


  
  #print(result)
  #print(len(result))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
