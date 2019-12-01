import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 13.3.2 生成器
class Generator(keras.Model):
    # 生成器网络
    def __init__(self):
        super(Generator, self).__init__()
        _filter = 64
        # 转置卷积层1，输出channel为filter*8，核大小4，步长1，不使用padding，不使用偏置
        self.conv1 = layers.Conv2DTranspose(_filter * 8, 4, 1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 转置卷积层2
        self.conv2 = layers.Conv2DTranspose(_filter * 4, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 转置卷积层3
        self.conv3 = layers.Conv2DTranspose(_filter * 2, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 转置卷积层4
        self.conv4 = layers.Conv2DTranspose(_filter * 1, 4, 2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 转置卷积层5
        self.conv5 = layers.Conv2DTranspose(3, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs  # [z, 100]
        # Reshape乘4D张量，方便后续转置卷积运算：(b, 1, 1, 100)
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        x = tf.nn.relu(x)  # 激活函数
        # 转置卷积-BN-激活函数：(b, 4, 4, 512)
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        # 转置卷积-BN-激活函数：(b, 8, 8, 256)
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        # 转置卷积-BN-激活函数：(b, 16, 16, 128)
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        # 转置卷积-BN-激活函数：(b, 32, 32, 64)
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        # 转置卷积-激活函数：(b, 64, 64, 3)
        x = self.conv5(x)
        x = tf.tanh(x)  # 输出x范围-1～1，与预处理一致

        return x


# 13.3.3 判别器
class Discriminator(keras.Model):
    # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()
        _filter = 64
        # 卷积层1
        self.conv1 = layers.Con2D(_filter, 4, 2, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 卷积层2
        self.conv2 = layers.Con2D(_filter * 2, 4, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 卷积层3
        self.conv3 = layers.Con2D(_filter * 4, 4, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 卷积层4
        self.conv4 = layers.Con2D(_filter * 8, 3, 1, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 卷积层5
        self.conv5 = layers.Con2D(_filter * 16, 3, 1, 'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        # 全局池化层
        self.pool = layers.GlobalAveragePooling2D()
        # 特征打平层
        self.flatten = layers.Flatten()
        # 2分类全连接层
        self.fc = layers.Dense(1)
