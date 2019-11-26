# TensorFlow basics
import tensorflow as tf

# 4.1.1 Data type
a = 1.2
aa = tf.constant(1.2)  # scalar
print(type(a), type(aa), tf.is_tensor(aa))

x = tf.constant([1, 2., 3.3])
print(x)
print(x.numpy())

a = tf.constant([1.2])
print(a, a.shape)

a = tf.constant([1, 2, 3.])
print(a, a.shape)

a = tf.constant([[1, 2], [3, 4]])
print(a, a.shape)

a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a)

# 4.1.2 String type
a = tf.constant('Hello, Deep Learning.')
print(a)

print(tf.strings.lower(a))

# 4.1.3 Bool type
a = tf.constant(True)
print(a)

a = tf.constant([True, False])
print(a)

a = tf.constant(True)
print(a == True)
