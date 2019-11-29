# 10.10 CIFAR10与VGG13实战
import tensorflow as tf
from tensorflow.keras import datasets


def preprocess(_x, _y):
    # [0~1]
    _x = 2 * tf.cast(_x, dtype=tf.float32) / 255. - 1
    _y = tf.cast(_y, dtype=tf.int32)
    return _x, _y


# 在线下载，加载CIFAR10数据集
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
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
