import tensorflow as tf

# 6.6 error
# 6.6.1 mean squared error
o = tf.random.normal([2, 10])
y_onehot = tf.constant([1, 3])
y_onehot = tf.one_hot(y_onehot, depth=10)
loss = tf.keras.losses.MSE(y_onehot, o)
print(loss)
loss = tf.reduce_mean(loss)
print(loss)

criteon = tf.keras.losses.MeanSquaredError()
loss = criteon(y_onehot, o)
print(loss)
