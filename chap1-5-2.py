# TensorFlow 2 vs 1.x

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

# 1. create computation graph
# create 2 inputs, specify the class and name
a_ph = tf.placeholder(tf.float32, name='variable_a')
b_ph = tf.placeholder(tf.float32, name='variable_b')
# create the op of the output, and name
c_op = tf.add(a_ph, b_ph, name='variable_c')
# 2. running computation graph
# create the running environment
sess = tf.InteractiveSession()
# initialize steps
init = tf.global_variables_initializer()
sess.run(init)  # run the initializer
# run the output, need to assign the values to the inputs
c_numpy = sess.run(c_op, feed_dict={a_ph: 2., b_ph: 4.})
# after the computation the output can get numerate type
print('a+b=', c_numpy)

import tensorflow as tf

# 1. create input tensor
a = tf.constant(2.)
b = tf.constant(4.)
# 2. directly compute and print
print('a+b=', a + b)
