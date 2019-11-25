import matplotlib
from matplotlib import pyplot as plt

# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# load MNIST dataset
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x_test:', x_test.shape, 'y_test:', y_test)

train_db = tf.data.Dataset.from_tensor_slices((x, y))

# 5.7.1 shuffle data
train_db = train_db.shuffle(1000)

# 5.7.2 batch training
batchsz = 512
train_db = train_db.batch(batchsz)


# 5.7.3 pre-processing
def preprocess(_x, _y):  # self-defined pre-processing function
    # call this func. would automatically load x, y with shape [b, 28, 28], [b]
    print(_x.shape, _y.shape)
    # normalized to 0 ~ 1
    _x = tf.cast(_x, dtype=tf.float32) / 255.
    _x = tf.reshape(_x, [-1, 28 * 28])  # flatten
    _y = tf.cast(_y, dtype=tf.int32)  # convert to tensor with int32
    _y = tf.one_hot(_y, depth=10)  # one-hot encoding

    return _x, _y


# the pre-processing function implemented in function preprocess
train_db = train_db.map(preprocess)
# iterate 20 epochs
train_db = train_db.repeat(20)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
x, y = next(iter(train_db))
print('train sample: ', x.shape, y.shape)
# print(x[0], y[0])


def main():

    lr = 1e-2   # learning rate
    accs, losses = [], []

    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros(256))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros(128))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros(10))

    for step, (x, y) in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as tape:

            # layer 1
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer 2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y - out)
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)
        
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)
        
        # print
        if step % 80 == 0:
            print(step, 'loss: ', float(loss))
            losses.append(float(loss))
        
        if step % 80 == 0:
            # evaluate / test
            total, total_correct = 0., 0

            for x, y in test_db:
                # layer 1
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer 2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]
            
            print(step, 'Evaluate Acc:', total_correct/total)

            accs.append(total_correct/total)
    
    plt.figure()
    x = [i*80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='Training')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
