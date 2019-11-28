import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets
import os

# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

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
    lr = 1e-2  # learning rate
    accs, losses = [], []

    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros(256))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros(128))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros(10))

    # 创建监控类，监控数据将写入log_dir目录
    log_dir = './log'
    summary_writer = tf.summary.create_file_writer(log_dir)

    for step, (_x, _y) in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        _x = tf.reshape(_x, [-1, 28 * 28])

        with tf.GradientTape() as tape:

            # layer 1
            h1 = _x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer 2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(_y - out)
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

            for _x_test, _y_test in test_db:
                # layer 1
                h1 = _x_test @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer 2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                _y_test = tf.argmax(_y_test, axis=1)
                # bool type
                correct = tf.equal(pred, _y_test)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += _x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)

            accs.append(total_correct / total)

            with summary_writer.as_default():
                # 当前时间戳step上的数据为loss，写入到ID位train-loss对象中
                tf.summary.scalar('train-loss', float(loss), step=step)
                # 写入测试准确率
                tf.summary.scalar('test-acc', float(total_correct/total), step=step)
                # 可视化测试用的图片，设置最多可视化9张图片
                # tf.summary.image("val-onebyone-images:", val_images, max_outputs=9, step=step)
                # 可视化真实标签直方图分布
                tf.summary.histogram('y-hist', y, step=step)
                # 查看文本信息
                tf.summary.text('loss-text', str(float(loss)), step=step)

    plt.figure()
    _x = [i * 80 for i in range(len(losses))]
    plt.plot(_x, losses, color='C0', marker='s', label='Training')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(_x, accs, color='C1', marker='s', label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Step')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
