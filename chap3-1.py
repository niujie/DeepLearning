# hand writing digits pictures dataset
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # convert to tensor and scale
y = tf.convert_to_tensor(y, dtype=tf.int32)  # convert to tensor
y = tf.one_hot(y, depth=10)  # one-hot coding
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # create data set class
train_dataset = train_dataset.batch(200)  # batch train

'''
# one-hot example
y = tf.constant([0, 1, 2, 3])   # number encoding
y = tf.one_hot(y, depth=10)     # one-hot encoding
print(y)
'''

model = keras.Sequential([  # 3 nested nonlinear layers
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    # Step 4. loop
    for step, (_x, _y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:  # create environment for recording gradient
            # flatten, [b, 28, 28] => [b, 784]
            _x = tf.reshape(_x, (-1, 28 * 28))
            # Step 1. obtain the model output
            # [b, 784] => [b, 10]
            out = model(_x)
            # Step 2. compute loss
            loss = tf.reduce_sum(tf.square(out - _y)) / _x.shape[0]

        # Step 3. calculate the gradients of the parameters w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad, update network parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())

    return loss.numpy()


def train():
    losses = []
    for epoch in range(50):
        loss = train_epoch(epoch)
        losses.append(loss)
    plt.plot(losses, marker='s')
    plt.show()


if __name__ == '__main__':
    train()
