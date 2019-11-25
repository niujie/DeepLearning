import tensorflow as tf

# 5.1 combine and slice
# 5.1.1 combine
a = tf.random.normal([4, 35, 8])  # simulate transcript A
b = tf.random.normal([6, 35, 8])  # simulate transcript B
print(tf.concat([a, b], axis=0))  # combine

a = tf.random.normal([10, 35, 4])
b = tf.random.normal([10, 35, 4])
print(tf.concat([a, b], axis=2))  # combine at different dimension

a = tf.random.normal([4, 32, 8])
b = tf.random.normal([6, 35, 8])
try:
    print(tf.concat([a, b], axis=0))  # illegal combine
except Exception as e:
    print('Exception: ', e)

a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
print(tf.stack([a, b], axis=0))  # stack combine

a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
print(tf.stack([a, b], axis=-1))  # stack combine at the end

a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
print(tf.concat([a, b], axis=0))  # concat combine, no extra dimension

a = tf.random.normal([35, 4])
b = tf.random.normal([35, 8])
try:
    print(tf.stack([a, b], axis=-1))  # illegal stack combine
except Exception as e:
    print('Exception: ', e)

# 5.1.2 slice
x = tf.random.normal([10, 35, 8])
# equally slice
result = tf.split(x, axis=0, num_or_size_splits=10)
print(len(result))
print(result[0])

x = tf.random.normal([10, 35, 8])
result = tf.split(x, axis=0, num_or_size_splits=[4, 2, 2, 2])
print(len(result))
print(result[0])

x = tf.random.normal([10, 35, 8])
result = tf.unstack(x, axis=0)  # unstack to length 1
print(len(result))
print(result[0])
