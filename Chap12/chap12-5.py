# 12.5 VAE实战
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


h_dim = 20
batchsz = 512
lr = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

z_dim = 10


# 12.5.1 VAE模型
class VAE(keras.Model):
    # 变分自编码器
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder网络
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)  # 均值输出
        self.fc3 = layers.Dense(z_dim)  # 方差输出

        # Decoder网络
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, _x):
        # 获得编码器的均值和方差
        h = tf.nn.relu(self.fc1(_x))
        # 均值
        _mu = self.fc2(h)
        # 方差的log
        _log_var = self.fc3(h)

        return _mu, _log_var

    def decoder(self, _z):
        # 根据隐藏变量z生成图片数据
        out = tf.nn.relu(self.fc4(_z))
        out = self.fc5(out)
        # 返回图片数据，784向量
        return out

    def call(self, inputs, training=None):
        # 前向计算
        # [b, 784] => [b, z_dim], [b, z_dim]
        _mu, _log_var = self.encoder(inputs)
        # reparameterization trick
        _z = reparameterize(_mu, _log_var)
        # 通过解码器生成
        _x_hat = self.decoder(_z)
        # 返回生成样本，及其均值和方差
        return _x_hat, _mu, _log_var


# 12.5.2 Reparameterization技巧
def reparameterize(_mu, _log_var):
    # reparameterize技巧，从正态分布采样epsilon
    eps = tf.random.normal(_log_var.shape)
    # 计算标准差
    std = tf.exp(_log_var) ** 0.5
    # reparameterize技巧
    _z = _mu + std * eps
    return _z


model = VAE()
model.build(input_shape=(4, 784))
optimizer = tf.optimizers.Adam(lr)

for epoch in range(100):  # 训练100个Epoch

    for step, x in enumerate(train_db):  # 遍历训练集
        # 打平，[b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        # 构建梯度记录器
        with tf.GradientTape() as tape:
            # 前向计算
            x_rec_logits, mu, log_var = model(x)
            # 损失计算
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
            # 计算KL散度N(mu, var) VS N(0, 1)
            # compute kl divergence (mu, var) ~ N (0, 1)
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]
            # 合并误差项
            loss = rec_loss + 1. * kl_div
        # 自动求导
        grads = tape.gradient(loss, model.trainable_variables)
        # 自动更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            # 打印训练误差
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))

    # evaluation
    # 测试生成效果，从正态分布随机采样z
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)   # 仅通过解码器生成图片
    x_hat = tf.sigmoid(logits)  # 转换为像素范围
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/sampled_epoch%d.png' % epoch)

    # 重建图片，从测试集采样图片
    x = next(iter(test_db))
    # 打平
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)
    x_hat = tf.sigmoid(x_hat_logits)    # 将输出转换为像素值
    # 恢复为28x28，[b, 784] => [b, 28, 28]
    x = tf.reshape(x, [-1, 28, 28])
    x_hat = tf.reshape(x_hat, [-1, 28, 28])
    # 输入的前50张+重建的前50张图片合并，[b, 28, 28] => [2b, 28, 28]
    x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
    x_concat = x_concat.numpy() * 255.  # 恢复0～255范围
    x_concat = x_concat.astype(np.uint8)
    save_images(x_concat, 'vae_images/rec_epoch%d.png' % epoch)
