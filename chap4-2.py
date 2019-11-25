import tensorflow as tf

# 4.2 Numerical accuracy
print(tf.constant(123456789, dtype=tf.int16))
print(tf.constant(123456789, dtype=tf.int32))

import numpy as np

print(np.pi)
print(tf.constant(np.pi, dtype=tf.float32))
print(tf.constant(np.pi, dtype=tf.float64))

# 4.2.1 reading accuracy
a = tf.constant(0.0, dtype=tf.float16)
print('before:', a.dtype)
if a.dtype != tf.float32:
    a = tf.cast(a, tf.float32)  # change the accuracy
print('after:', a.dtype)

# 4.2.2 change the type
a = tf.constant(np.pi, dtype=tf.float16)
print(tf.cast(a, tf.double))

a = tf.constant(123456789, dtype=tf.int32)
print(tf.cast(a, tf.int16))

a = tf.constant([True, False])
print(tf.cast(a, tf.int32))

a = tf.constant([-1, 0, 1, 2])
print(tf.cast(a, tf.bool))
