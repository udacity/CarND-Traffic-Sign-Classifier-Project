# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Hyperparameters
MEAN = 0.0
STDDEV = 0.01

def conv_relu_layer(x, filter_w, in_d, out_d):
    conv_W = tf.Variable(tf.truncated_normal(shape = (filter_w, filter_w, in_d, out_d), mean = MEAN, stddev = STDDEV))
    conv_b = tf.Variable(tf.zeros(out_d))
    conv_res = tf.nn.conv2d(x, conv_W, strides=[1, 1, 1, 1], padding = 'VALID') + conv_b
    return tf.nn.relu(conv_res)

def maxpool_stride_layer(x, filter_w, s):
    return tf.nn.max_pool(x, ksize = [1, filter_w, filter_w, 1], strides = [1, s, s, 1], padding = 'VALID')

def fc_relu_dropout_layer(x, a, b, keep_prob):
    fc_W = tf.Variable(tf.truncated_normal(shape=(a, b), mean = MEAN, stddev = STDDEV))
    fc_b = tf.Variable(tf.zeros(b))
    fc = tf.matmul(x, fc_W) + fc_b
    fc_relu = tf.nn.relu(fc)
    return tf.nn.dropout(fc_relu, keep_prob)

def logits(x, a, b):
    fc_W = tf.Variable(tf.truncated_normal(shape = (a, b), mean = MEAN, stddev = STDDEV))
    fc_b = tf.Variable(tf.zeros(b))
    return tf.matmul(x, fc_W) + fc_b

def net(x, keep_prob, out):
    conv0 = conv_relu_layer(x, filter_w = 2, in_d = 3, out_d = 6)
    conv1 = conv_relu_layer(conv0, filter_w = 5, in_d = 6, out_d = 16)
    maxpool1 = maxpool_stride_layer(conv1, 2, 2)
    fl = flatten(maxpool1)
    fc1 = fc_relu_dropout_layer(fl, fl.get_shape().as_list()[-1], 1024, keep_prob)
    fc2 = fc_relu_dropout_layer(fc1, fc1.get_shape().as_list()[-1], 1024, keep_prob)
    return logits(fc2, fc2.get_shape().as_list()[-1], out)
