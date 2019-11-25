import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal([4, 28*28])

# 6.3.1 tensor
# layer 1
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# layer 2
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# layer 3
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# output layer
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

with tf.GradientTape() as tape:
    # x: [b, 28*28]
    # layer 1 forward cal., [b, 28*28] => [b, 256]
    h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)
    # layer 2 forward cal., [b, 256] => [b, 128]
    h2 = h1 @ w2 + b2
    h2 = tf.nn.relu(h2)
    # layer 3 forward cal., [b, 128] => [b, 64]
    h3 = h2 @ w3 + b3
    h3 = tf.nn.relu(h3)
    # output layer forward cal., [b, 64] => [b, 10]
    h4 = h3 @ w4 + b4

# 6.3.2 layer
fc1 = layers.Dense(256, activation=tf.nn.relu)  # layer 1
fc2 = layers.Dense(128, activation=tf.nn.relu)  # layer 2
fc3 = layers.Dense(64, activation=tf.nn.relu)  # layer 3
fc4 = layers.Dense(10, activation=tf.nn.relu)  # output layer

h1 = fc1(x)
h2 = fc2(h1)
h3 = fc3(h2)
h4 = fc4(h3)

# network class using Sequential container
model = tf.keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=None)
])
out = model(x)  # forward cal.
