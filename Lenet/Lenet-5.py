import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

mnist = input_data.read_data_sets("MNIST/", one_hot=True)
batch_size = 128
epochs = 100
with tf.variable_scope("input"):
    x = tf.placeholder(shape=(None, 784), dtype=tf.float32, name="input_x")
    y = tf.placeholder("float", name="input_y")

def lenet(input):
    with tf.variable_scope("reshape"):
        input = tf.reshape(input, [-1, 28, 28, 1], name="reshape_input_x")

    with tf.variable_scope("conv_1"):
        weights1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0, stddev=0.1), name="weights1")
        bias1 = tf.Variable(tf.truncated_normal(shape=[6], mean=0, stddev=0.1), name="bias1")
        c1 = tf.nn.conv2d(input=input, filter=weights1, strides=[1,1,1,1], padding="SAME")+bias1
    with tf.variable_scope("max_pool_1"):
        s2 = tf.nn.max_pool(c1, ksize=(1,2,2,1),strides=[1,2,2,1], padding="VALID")

    with tf.variable_scope("conv_2"):
        weights2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.1), name="weights2")
        bias2 = tf.Variable(tf.truncated_normal(shape=[16]))
        c3 = tf.nn.conv2d(input=s2, filter=weights2, strides=[1,1,1,1], padding="VALID")+bias2
    with tf.variable_scope("max_pool_2"):
        s4 = tf.nn.max_pool(c3, ksize=(1,2,2,1),strides=[1,2,2,1], padding="VALID")

    with tf.variable_scope("conv_3"):
        weights3 = tf.Variable(tf.truncated_normal(shape=[5,5,16,120], mean=0, stddev=0.1))
        bias3 = tf.Variable(tf.truncated_normal(shape=[120], mean=0, stddev=0.1))
        c5 = tf.nn.conv2d(input=s4, filter=weights3, strides=[1,1,1,1], padding="VALID")+bias3

    c5_shape_li = c5.get_shape().as_list()
    with tf.variable_scope("fc_1"):
        weights4 = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=0, stddev=0.1))
        bias4 = tf.Variable(tf.truncated_normal(shape=[84], mean=0, stddev=0.1))
        with tf.variable_scope("flatten"):
            o5_flatten = tf.reshape(tensor=c5, shape=[-1, c5_shape_li[1] * c5_shape_li[2] * c5_shape_li[3]], name="flatten")
        f6 = tf.matmul(o5_flatten, weights4) + bias4

    with tf.variable_scope("output"):
        weights5 = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=0, stddev=0.1))
        bias5 = tf.Variable(tf.truncated_normal(shape=[10], mean=0, stddev=0.1))
        output = tf.matmul(f6, weights5) + bias5

    return output

predict = lenet(x)
with tf.variable_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))

with tf.variable_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    correct_train = tf.equal(tf.arg_max(predict, 1), tf.arg_max(y, 1))
    accuracy_train = tf.reduce_mean(tf.cast(correct_train, "float"))

with tf.Session() as sess:
    saver = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in tqdm(range(10)):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
        print("accuracy", sess.run(accuracy_train, feed_dict={x: mnist.test.images, y:mnist.test.labels}),
              'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(predict, -1), tf.argmax(y, -1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
