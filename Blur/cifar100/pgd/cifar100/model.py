"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 1024])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    epsilon=.001
    beta=1000
    self.x_image = tf.reshape(self.x_input, [-1, 32, 32, 1])
    
    mean,var=tf.nn.moments(self.x_image, axes=[0])
    gamma=tf.Variable(tf.ones([1]))
    beta=tf.Variable(tf.zeros([1]))
    batch_norm=tf.nn.batch_normalization(self.x_image,mean,var,beta,gamma,epsilon)
    
    # first convolutional layer
    W_conv1 = self._weight_variable([3,3,1,100])
    b_conv1 = self._bias_variable([100])
    net_out1=self._conv2d(batch_norm, W_conv1) + b_conv1
    
    mean1,var1=tf.nn.moments(net_out1, axes=[0])
    gamma1=tf.Variable(tf.ones([100]))
    beta1=tf.Variable(tf.zeros([100]))
    batch_norm1=tf.nn.batch_normalization(net_out1,mean1,var1,beta1,gamma1,epsilon)
    
    h_conv1 = tf.nn.relu(batch_norm1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([3,3,100,64])
    b_conv2 = self._bias_variable([64])
    net_out2=self._conv2d(h_pool1, W_conv2) + b_conv2
    
    mean2,var2=tf.nn.moments(net_out2, axes=[0])
    gamma2=tf.Variable(tf.ones([64]))
    beta2=tf.Variable(tf.zeros([64]))
    batch_norm2=tf.nn.batch_normalization(net_out2,mean2,var2,beta2,gamma2,epsilon)
    
    h_conv2 = tf.nn.relu(batch_norm2)
    h_pool2 = self._max_pool_2x2(h_conv2)         
    # final convolutional layer
    W_conv_f = self._weight_variable([5,5,64,64])
    
    b_conv_f = self._bias_variable([64])
    net_out_f=self._conv2d(h_pool2, W_conv_f) + b_conv_f
    
    mean_f,var_f=tf.nn.moments(net_out_f, axes=[0])
    gamma_f=tf.Variable(tf.ones([64]))
    beta_f=tf.Variable(tf.ones([64])/beta)
    batch_norm_f=tf.nn.batch_normalization(net_out_f,mean_f,var_f,beta_f,gamma_f,epsilon)
    
    h_pool_f = tf.nn.relu(batch_norm_f)
    
    
    
    # first fully connected layer
    W_fc1 = self._weight_variable([4096, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool_flat = tf.reshape(h_pool_f, [-1, 4096])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
    
    # second fully connected layer
    W_fc2 = self._weight_variable([1024, 512])
    b_fc2 = self._bias_variable([512])

    h_fc3 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    # output layer
    W_fc3 = self._weight_variable([512,100])
    b_fc3 = self._bias_variable([100])

    self.pre_softmax = tf.matmul(h_fc3, W_fc3) + b_fc3

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
