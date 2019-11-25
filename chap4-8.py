import tensorflow as tf

# 4.8 broadcasting
A = tf.random.normal([32, 1])
print(tf.broadcast_to(A, [2, 32, 32, 3]))

A = tf.random.normal([32, 2])
try:
    print(tf.broadcast_to(A, [2, 32, 32, 4]))
except Exception as e:
    print('except: ', e)
