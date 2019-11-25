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

# 5.6.3 tf.boolean_mask
print(tf.boolean_mask(x, mask=[True, False, False, True], axis=0))
print(tf.boolean_mask(x, mask=[True, False, False, True, True, False, False, True], axis=2))

x = tf.random.uniform([2, 3, 8], maxval=100, dtype=tf.int32)
print(tf.gather_nd(x, [[0, 0], [0, 1], [1, 1], [1, 2]]))    # multi-dim pos sampling
print(tf.boolean_mask(x, [[True, True, False], [False, True, True]]))   # multi-dim mask sampling

# 5.6.4 tf.where
a = tf.ones([3, 3])   # create a with all ones
b = tf.zeros([3, 3])  # create b with all zeros
# create sampling mask
cond = tf.constant([[True, False, False], [False, True, False], [True, True, False]])
print(tf.where(cond, a, b))
print(cond)
print(tf.where(cond))  # obtain pos of True in cond
