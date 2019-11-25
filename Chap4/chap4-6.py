import tensorflow as tf

# 4.6 index and slice
# 4.6.1 index
# create 4 32x32 color image input
x = tf.random.normal([4, 32, 32, 3])

print(x[0])
print(x[0][1])
print(x[0][1][2])
print(x[2][1][0][1])
print(x[1, 9, 2])

# 4.6.2 slice
print(x[1:3])
print(x[0, ::])
print(x[:, 0:28:2, 0:28:2, :])

x = tf.range(9)
print(x[8:0:-1])
print(x[::-1])
print(x[::-2])

x = tf.random.normal([4, 32, 32, 3])
print(x[0, ::-2, ::-2])
print(x[:, :, :, 1])
print(x[0:2, ..., 1:])
print(x[2:, ...])
print(x[..., :2])
