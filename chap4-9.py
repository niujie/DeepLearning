import tensorflow as tf

# 4.9 Math
# 4.9.1 add, subtract, multiply, division
a = tf.range(5)
b = tf.constant(2)
print(a//b)
print(a%b)

# 4.9.2 power
x = tf.range(4)
print(tf.pow(x, 3))

x = tf.constant([1., 4., 9.])
print(x ** 0.5)

x = tf.range(5)
x = tf.cast(x, dtype=tf.float32)
x = tf.square(x)
print(x)
print(tf.sqrt(x))

# 4.9.3 exponential and log
x = tf.constant([1., 2., 3.])
print(2 ** x)
print(tf.exp(1.))

x = tf.exp(3.)
print(tf.math.log(x))

x = tf.constant([1., 2.])
x = 10 ** x
print(tf.math.log(x) / tf.math.log(10.))

# 4.9.4 matrix multiplication
a = tf.random.normal([4, 3, 28, 32])
b = tf.random.normal([4, 3, 32, 2])
print(a@b)

a = tf.random.normal([4, 28, 32])
b = tf.random.normal([32, 16])
print(tf.matmul(a, b))
