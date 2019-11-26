import tensorflow as tf

# 5.5 limit the data
x = tf.range(9)
print(tf.maximum(x, 2))
print(tf.minimum(x, 7))


def relu(_x):
    return tf.minimum(_x, 0.)  # limit the lower range as 0


print(tf.minimum(tf.maximum(x, 2), 7))  # limit to 2 ~ 7
print(tf.clip_by_value(x, 2, 7))        # limit to 2 ~ 7
