# -*- coding: utf-8 -*-
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/", one_hot=True)

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32,[None,784])
    y_true = tf.placeholder(tf.float32,[None,10])

    x_image = tf.reshape(x, [-1,28,28,1])
'''
def weight_variable(shape):
    with tf.name_scope('weights'):
        W = tf.Variable(
        tf.truncated_normal(shape, stddev=0.1),
        name='W')
    return W

def bias_variable(shape):
    with tf.name_scope('biases'):
        b = tf.Variable(
        tf.constant(0.1, shape=shape),
        name='b')
    return b

def conv2d(x, W, b, activation_function=None, name='conv'):
    #with tf.name_scope(name):
    with tf.variable_scope(name):
        y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

        if activation_function is None:
            y = y + b
        else:
            y = activation_function((y+b), )
    return y

def max_pool_2x2(x, name='max_pool'):
    #with tf.name_scope(name):
    with tf.variable_scope(name):
        y = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return y

def fully_connect(x, W, b, activation_function=None, name='fully_connect'):
    #with tf.name_scope(name):
    with tf.variable_scope(name):
        y = tf.matmul(x, W) + b

        if activation_function is None:
            y = y + b
        else:
            y = activation_function((y+b), )

    return y

'''
def weight_variable(shape, name='W'):
    W = tf.Variable(
        tf.truncated_normal(shape, stddev=0.1),
        name=name)
    return W

def bias_variable(shape, name='b'):
    b = tf.Variable(
        tf.constant(0.1, shape=shape),
        name=name)
    return b

def conv2d(x, W, b, activation_function=None):
    y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    if activation_function is None:
        y = y + b
    else:
        y = activation_function((y+b), )
    return y

def max_pool_2x2(x, name='max_pool'):
    y = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return y

def fully_connect(x, W, b, activation_function=None):
    y = tf.matmul(x, W) + b

    if activation_function is None:
        y = y + b
    else:
        y = activation_function((y+b), )
    return y

# The first convolutional layer
with tf.variable_scope('conv_1'):
    W_conv1 = weight_variable([5,5,1,32], 'W_conv1')                               
    b_conv1 = bias_variable([32], 'b_conv1')
    h_conv1 = conv2d(x_image, W_conv1, b_conv1, tf.nn.relu)

with tf.variable_scope('max_pool_1'):
    h_pool1 = max_pool_2x2(h_conv1)

# The second convolutional layer
with tf.variable_scope('conv_2'):
    W_conv2 = weight_variable([5,5,32,64], 'W_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')
    h_conv2 = conv2d(h_pool1, W_conv2, b_conv2, tf.nn.relu)

with tf.variable_scope('max_pool_2'):
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])

# The first fully connected layer
with tf.variable_scope('fully_connect_1'):
    W_fc1 = weight_variable([7*7*64,1024], 'W_fc1')
    b_fc1 = bias_variable([1024], 'b_fc1')
 
    h_fc1 = fully_connect(h_pool2_flat, W_fc1, b_fc1, tf.nn.relu)
 
    # Use dropout in first connected layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# The second fully connected layer output the prediction
with tf.variable_scope('fully_connect_2'):
    W_fc2 = weight_variable([1024,10], 'W_fc2')
    b_fc2 = bias_variable([10], 'b_fc2')
#y_pre = fully_connect(h_fc1_drop, W_fc2, b_fc2, tf.nn.softmax)
# You can define the cross entropy 
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_fc2), reduction_indices=[1]))
    logits = fully_connect(h_fc1_drop, W_fc2, b_fc2, None)

with tf.variable_scope('loss'):
    y_pre = tf.nn.softmax(logits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

with tf.variable_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.variable_scope('accuracy'):
    correct_predition = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_true,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # tf.train.SummaryWriter soon be deprecated, use following
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    for i in range(20000):
        batch = mnist.train.next_batch(128)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_true:batch[1], keep_prob:1.0})
            print("step %d,training accuracy %g"%(i, train_accuracy))
    
        sess.run(train_step, feed_dict={x:batch[0], y_true:batch[1], keep_prob:0.5})
    
    accu = sess(accuracy, feed_dict={x:mnist.test.images, y_true:mnist.test.labels, keep_prob:1.0})
    print("test accuracy %g"%accu)
