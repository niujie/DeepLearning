import tensorflow as tf
from tensorflow.keras import layers, Sequential, datasets, optimizers
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
    # 将数据映射到-1~1
    _x = 2 * tf.cast(_x, dtype=tf.float32) / 255. - 1
    _y = tf.cast(_y, dtype=tf.int32)  # 类型转换
    return _x, _y


(x, y), (x_test, y_test) = datasets.cifar10.load_data()  # 加载数据集
y = tf.squeeze(y, axis=1)  # 删除不必要的维度
y_test = tf.squeeze(y_test, axis=1)  # 删除不必要的维度
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 构建训练集
# 随机打散，预处理，批量化
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 构建测试集
# 预处理，批量化
test_db = test_db.map(preprocess).batch(128)

# 采样一个样本
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


# 10.12 深度残差网络
# 10.12.2 ResBlock实现
class BasicBlock(layers.Layer):
    # 残差模块类
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # f(x)包含了2个普通卷积层，创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 创建卷积层2
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:  # 插入identity层
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:  # 否则，直接连接
            self.downsample = lambda _x: _x

    def call(self, inputs, training=None):
        # 前向传播函数
        out = self.conv1(inputs)  # 通过第一个卷积层
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 通过第二个卷积层
        out = self.bn2(out)
        # 输入通过identity()转换
        identity = self.downsample(inputs)
        # f(x)+x运算
        output = layers.add([out, identity])
        # 再通过激活函数并返回
        output = tf.nn.relu(output)
        return output


# 10.14 CIFAR10与ResNet18实战
def build_resblock(filter_num, blocks, stride=1):
    # 辅助函数，堆叠filter_num个BasicBlock
    res_blocks = Sequential()
    # 只有第一个BasicBlock的步长可能不为1，实现下采样
    res_blocks.add(BasicBlock(filter_num, stride))

    for _ in range(1, blocks):  # 其它BasicBlock步长都为1
        res_blocks.add(BasicBlock(filter_num, stride=1))

    return res_blocks


class ResNet(tf.keras.Model):
    # 通用的ResNet实现类
    def __init__(self, layer_dims, num_classes=10):  # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        # 根网络，预处理
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 堆叠4个Block，每个block包含了多个BasicBlock，设置步长不一样
        self.layer1 = build_resblock(64, layer_dims[0])
        self.layer2 = build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = build_resblock(512, layer_dims[3], stride=2)

        # 通过Pooling层将高宽降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # 通过根网络
        _x = self.stem(inputs)
        # 依次通过4个模块
        _x = self.layer1(_x)
        _x = self.layer2(_x)
        _x = self.layer3(_x)
        _x = self.layer4(_x)

        # 通过池化层
        _x = self.avgpool(_x)
        # 通过全连接层
        _x = self.fc(_x)

        return _x


def resnet18():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([2, 2, 2, 2])


def resnet34():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([3, 4, 6, 3])


def main():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()  # ResNet18网络
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()  # 统计网络参数
    optimizer = optimizers.Adam(lr=1e-4)  # 构建优化器

    for epoch in range(50):  # 训练epoch

        for step, (_x, _y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 10]，前向传播
                logits = model(_x)
                # [b] => [b, 10]，one-hot编码
                y_onehot = tf.one_hot(_y, depth=10)
                # 计算交叉熵
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            # 计算梯度信息
            grads = tape.gradient(loss, model.trainable_variables)
            # 更新网络参数
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        for _x, _y in test_db:
            logits = model(_x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, _y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += _x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
