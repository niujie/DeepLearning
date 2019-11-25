import tensorflow as tf

# 4.7 Convert dimension
# 4.7.1 reshape
x = tf.range(96)
x = tf.reshape(x, [2, 4, 4, 3])
print(x)
print(x.ndim, x.shape)
print(tf.reshape(x, [2, -1]))
print(tf.reshape(x, [2, 4, 12]))
print(tf.reshape(x, [2, -1, 3]))

# 4.7.2 increase or reduce dimension
x = tf.random.uniform([28, 28], maxval=10, dtype=tf.int32)
print(x)
x = tf.expand_dims(x, axis=2)
print(x)
x = tf.expand_dims(x, axis=0)
print(x)
x = tf.squeeze(x, axis=0)
print(x)
x = tf.squeeze(x, axis=2)
print(x)
x = tf.random.uniform([1, 28, 28, 1], maxval=10, dtype=tf.int32)
print(tf.squeeze(x))

# 4.7.3 change dimension
x = tf.random.normal([2, 32, 32, 3])
print(tf.transpose(x, perm=[0, 3, 1, 2]))
x = tf.random.normal([2, 32, 32, 3])
print(tf.transpose(x, perm=[0, 2, 1, 3]))

# 4.7.4 copy data
b = tf.constant([1, 2])
b = tf.expand_dims(b, axis=0)
print(b)
b = tf.tile(b, multiples=[2, 1])
print(b)

x = tf.range(4)
x = tf.reshape(x, [2, 2])
print(x)
x = tf.tile(x, multiples=[1, 2])
print(x)
x = tf.tile(x, multiples=[2, 1])
print(x)
