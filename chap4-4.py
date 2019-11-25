import tensorflow as tf
import numpy as np

# 4.4 create tensor
# 4.4.1 from Numpy, List classes
print(tf.convert_to_tensor([1, 2.]))

print(tf.convert_to_tensor(np.array([[1, 2.], [3, 4]])))

# 4.4.2 create tensor with all zeros or all ones
print(tf.zeros([]), tf.ones([]))

print(tf.zeros([1]), tf.ones([1]))

print(tf.zeros([2, 2]))

print(tf.ones([3, 2]))

a = tf.ones([2, 3])
print(tf.zeros_like(a))

a = tf.zeros([3, 2])
print(tf.ones_like(a))

# 4.4.3 create self-defined numeric tensor
print(tf.fill([], -1))

print(tf.fill([1], -1))

print(tf.fill([2, 2], 99))

# 4.4.4 create tensor with known distribution
print(tf.random.normal([2, 2]))

print(tf.random.normal([2, 2], mean=1, stddev=2))

print(tf.random.uniform([2, 2]))

print(tf.random.uniform([2, 2], maxval=10))

print(tf.random.uniform([2, 2], maxval=100, dtype=tf.int32))

# 4.4.5 create sequence
print(tf.range(10))

print(tf.range(10, delta=2))

print(tf.range(1, 10, delta=2))
