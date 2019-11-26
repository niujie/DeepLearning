import tensorflow as tf

# 7.6 chain rule
# variables to be optimized
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)
# gradient record
with tf.GradientTape(persistent=True) as tape:
    # the gradient information for non tf.Variable tensor needs to be tracked manually
    tape.watch([w1, b1, w2, b2])
    # create 2 linear layers
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2

# cal. partial derivatives individually
dy2_dy1 = tape.gradient(y2, [y1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]
dy2_dw1 = tape.gradient(y2, [w1])[0]

# check the chain rule
print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)
