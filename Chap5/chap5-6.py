import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
print(tf.gather_nd(x, [[0, 0], [0, 1], [1, 1], [1, 2]]))  # multi-dim pos sampling
print(tf.boolean_mask(x, [[True, True, False], [False, True, True]]))  # multi-dim mask sampling

# 5.6.4 tf.where
a = tf.ones([3, 3])  # create a with all ones
b = tf.zeros([3, 3])  # create b with all zeros
# create sampling mask
cond = tf.constant([[True, False, False], [False, True, False], [True, True, False]])
print(tf.where(cond, a, b))
print(cond)
print(tf.where(cond))  # obtain pos of True in cond

x = tf.random.normal([3, 3])
print(x)
mask = x > 0
print(mask)
indices = tf.where(mask)
print(indices)
print(tf.gather_nd(x, indices))
print(tf.boolean_mask(x, mask))

# 5.6.5 scatter_nd
# create the pos to be refreshed
indices = tf.constant([[4], [3], [1], [7]])
# create data
updates = tf.constant([4.4, 3.3, 1.1, 7.7])
# write in updates on vector with all zeros and length 8 according to indices
print(tf.scatter_nd(indices, updates, [8]))

# create the pos for written in
indices = tf.constant([[1], [3]])
# create data for written in
updates = tf.constant([
    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
])
# write in updates on a [4, 4, 4] tensor according to indices
print(tf.scatter_nd(indices, updates, [4, 4, 4]))

# 5.6.6 meshgrid
x = tf.linspace(-8., 8., 100)
y = tf.linspace(-8., 8., 100)
x, y = tf.meshgrid(x, y)
print(x.shape, y.shape)
z = tf.sqrt(x**2 + y**2)
z = tf.sin(z) / z

fig = plt.figure()
ax = Axes3D(fig)
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), levels=100)
plt.show()
