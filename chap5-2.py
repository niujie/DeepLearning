import tensorflow as tf
from tensorflow import keras
import numpy as np

# 5.2 statistics
# 5.2.1 vector norm
x = tf.ones([2, 2])
print(tf.norm(x, ord=1))    # L1 norm
print(tf.norm(x, ord=2))    # L2 norm
print(tf.norm(x, ord=np.inf))   # \inf norm

# 5.2.2 max, min, mean, sum
x = tf.random.normal([4, 10])
print(tf.reduce_max(x, axis=1))  # max
print(tf.reduce_min(x, axis=1))  # min
print(tf.reduce_mean(x, axis=1))    # mean

# global max, min, mean, sum
print(tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x))

out = tf.random.normal([4, 10])  # network predicted output
y = tf.constant([1, 2, 2, 0])    # true label
y = tf.one_hot(y, depth=10)      # one=hot encoding
loss = keras.losses.mse(y, out)  # error for each sample
loss = tf.reduce_mean(loss)      # mean error
print(loss)

out = tf.random.normal([4, 10])
print(tf.reduce_sum(out, axis=-1))  # sum

out = tf.random.normal([2, 10])
out = tf.nn.softmax(out, axis=1)    # convert to probability
print(out)
pred = tf.argmax(out, axis=1)   # position of the max probability
print(pred)
