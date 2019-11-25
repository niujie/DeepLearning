import tensorflow as tf

# 4.3 optimizable tensor
a = tf.constant([-1, 0, 1, 2])
aa = tf.Variable(a)
print(aa.name, aa.trainable)

a = tf.Variable([[1, 2], [3, 4]])
print(a)
