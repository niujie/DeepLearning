import tensorflow as tf

# 5.4 fill and copy
# 5.4.1 fill
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([7, 8, 1, 6])
b = tf.pad(b, [[0, 2]])  # padding
print(b)
print(tf.stack([a, b], axis=0))  # combine

total_words = 10000  # size of the vocabulary
max_review_len = 80  # max length of the sentence
embedding_len = 100  # word vector length
# load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
# fill or cut at the end of the sentences to same length
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len, truncating='post',
                                                        padding='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len, truncating='post', padding='post')
print(x_train.shape, x_test.shape)

x = tf.random.normal([4, 28, 28, 1])
# fill 2 pixels for up, down, left, right of the image
print(tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]]))

# 5.4.2 copy
x = tf.random.normal([4, 32, 32, 3])
print(tf.tile(x, [2, 3, 3, 1]))
