import tensorflow as tf
from tensorflow.keras import layers

# 6.2.1 tensor
x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1  # linear transform
o1 = tf.nn.relu(o1)         # activation func.
print(o1)

# 6.2.2 layer
x = tf.random.normal([4, 28*28])
# create fully connected layer, specify the number of output nodes and activation func.
fc = layers.Dense(512, activation=tf.nn.relu)
h1 = fc(x)  # perform the calculation on fully connected layer via fc
print(h1)
print(fc.kernel)    # weight matrix of Dense layer
print(fc.bias)      # bias vector of Dense layer
print(fc.trainable_variables)
print(fc.variables)
