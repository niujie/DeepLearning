import tensorflow as tf

# 6.5.3 [0, 1] and sum as 1
z = tf.constant([2., 1., 0.1])
print(tf.nn.softmax(z))

z = tf.random.normal([2, 10])
y_onehot = tf.constant([1, 3])
y_onehot = tf.one_hot(y_onehot, depth=10)
# output layer without Softmax func., so set from_logits = True
loss = tf.keras.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
loss = tf.reduce_mean(loss)
print(loss)

# create Softmax and CrossEntropy cal. class, z from output layer with softmax
criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot, z)
print(loss)

# 6.5.4 [-1, 1]
x = tf.linspace(-6., 6., 10)
print(tf.tanh(x))
