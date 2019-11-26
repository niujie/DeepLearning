import tensorflow as tf

# 5.3 compare tensors
out = tf.random.normal([100, 10])
out = tf.nn.softmax(out, axis=1)  # convert to probability
pred = tf.argmax(out, axis=1)  # max position
print(pred)

# true tag
y = tf.random.uniform([100], dtype=tf.int64, maxval=10)
print(y)

out = tf.equal(pred, y)  # compare prediction and true tag
print(out)

out = tf.cast(out, dtype=tf.float32)
correct = tf.reduce_sum(out)  # count the true values
print(correct)
