import tensorflow as tf

# 5.6 advanced operation
# 5.6.1 tf.gather
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)
print(tf.gather(x, [0, 1], axis=0))  # collect x[0~1,:,:], same as x[:2]
print(tf.gather(x, [0, 3, 8, 11, 12, 26], axis=1))
print(tf.gather(x, [2, 4], axis=2))

a = tf.range(8)
a = tf.reshape(a, [4, 2])  # create tensor
print(a)
print(tf.gather(a, [3, 1, 0, 2], axis=0))

students = tf.gather(x, [1, 2], axis=0)
print(students)
print(tf.gather(students, [2, 3, 5, 26], axis=1))

print(x[1, 1])
print(tf.stack([x[1, 1], x[2, 2], x[3, 3]], axis=0))

# 5.6.2 tf.gather_nd
print(tf.gather_nd(x, [[1, 1], [2, 2], [3, 3]]))
print(tf.gather_nd(x, [[1, 1, 2], [2, 2, 3], [3, 3, 4]]))
