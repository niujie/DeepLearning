# 10.10 CIFAR10与VGG13实战
import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# 获取所有GPU设备列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUS,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)


def preprocess(_x, _y):
    # [0~1]
    _x = 2 * tf.cast(_x, dtype=tf.float32) / 255. - 1
    _y = tf.cast(_y, dtype=tf.int32)
    return _x, _y


# 在线下载，加载CIFAR10数据集
(x, y), (x_test, y_test) = datasets.cifar10.load_data()
# 删除y的一个维度,[b, 1] => [b]
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
# 打印训练集和测试集的形状
print(x.shape, y.shape, x_test.shape, y_test.shape)
# 构建训练集对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)
# 构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(128)
# 从训练集中采样一个Batch，并观察
sample = next(iter(train_db))
print('sample: ', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

conv_layers = [  # 先创建包含多层的列表
    # Conv-Conv-Pooling单元1
    # 64个3x3卷积核，输入输出同大小
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 高宽减半
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元2，输出通道提升至128，高宽大小减半
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元3，输出通道提升至256，高宽大小减半
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元4，输出通道提升至512，高度大小减半
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元5，输出通道提升至512，高度大小减半
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
]


def main():
    # 利用前面创建的层列表构建网络容器
    conv_net = Sequential(conv_layers)

    # 创建3层全连接层子网络
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None)
    ])

    # build 2个子网络，并打印网络参数信息
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    conv_net.summary()
    fc_net.summary()
    optimizer = optimizers.Adam(lr=1e-4)

    # 列表合并，合并2个子网络的参数
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(50):

        for step, (_x, _y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = conv_net(_x)
                # flatten, => [b, 512]
                out = tf.reshape(out, [-1, 512])
                # [b, 512] => [b, 10]
                logits = fc_net(out)
                # [b] => [b, 10]
                y_onehot = tf.one_hot(_y, depth=10)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            # 对所有参数求梯度
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        for _x, _y in test_db:
            out = conv_net(_x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=-1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, _y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += _x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
