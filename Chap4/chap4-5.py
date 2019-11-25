import tensorflow as tf

# 4.5.1 Scalar
out = tf.random.uniform([4, 10])  # randomly simulate network output
y = tf.constant([2, 3, 2, 0])  # randomly create the real tag for samplings
y = tf.one_hot(y, depth=10)  # one-hot encoding
loss = tf.keras.losses.mse(y, out)  # calculate MSE for each sample
loss = tf.reduce_mean(loss)  # averaged MSE
print(loss)

# 4.5.2 Vector
# z = wx, simulate to obtain the input z for the activation function
z = tf.random.normal([4, 2])
b = tf.zeros([2])  # simulate the bias
z = z + b  # add up the bias
print(z)

from tensorflow.keras import layers

fc = layers.Dense(3)  # create one layer Wx+b, with 3 output nodes
# build W, b tensors, with 4 input nodes
fc.build(input_shape=(2, 4))
print(fc.bias)  # check the bias

# 4.5.3 Matrix
x = tf.random.normal([2, 4])
w = tf.ones([4, 3])  # define tensor W
b = tf.zeros([3])  # define tensor b
o = x @ w + b  # X@W+b operation
print(o)

fc = layers.Dense(3)  # define all-connected layer with 3 output nodes
fc.build(input_shape=(2, 4))  # define all-connected layer with 4 input nodes
print(fc.kernel)

# 4.5.4 3D tensor
# auto load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
# fill and cut the sentences into pieces with 80 words
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)
print(x_train.shape)
# create words vector Embedding layer
embedding = layers.Embedding(10000, 100)
# convert the digitally encoded words to words vector
out = embedding(x_train)
print(out.shape)

# 4.5.5 4D tensor
# create 4 32x32 color image input
x = tf.random.normal([4, 32, 32, 3])
# create CNN
layer = layers.Conv2D(16, kernel_size=3)
out = layer(x)  # forward compute
print(out.shape)
print(layer.kernel.shape)
