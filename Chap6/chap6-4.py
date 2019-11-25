import tensorflow as tf

# 6.4.1 Sigmoid
x = tf.linspace(-6., 6., 10)
print(x)
print(tf.nn.sigmoid(x))

# 6.4.2 ReLU
print(tf.nn.relu(x))

# 6.4.3 LeakyReLU
print(tf.nn.leaky_relu(x, alpha=0.1))

# 6.4.4 Tanh
print(tf.nn.tanh(x))
