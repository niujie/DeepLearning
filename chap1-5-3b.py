# auto gradient
import tensorflow as tf

# create 4 tensor
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape:  # create gradient environment
    tape.watch([w])  # add w to gradient trace list
    # compute process
    y = a * w ** 2 + b * w + c

# calculate derivative
[dy_dw] = tape.gradient(y, [w])
print(dy_dw)  # print derivative
