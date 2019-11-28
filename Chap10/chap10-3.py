import tensorflow as tf
from tensorflow.keras import layers

# 10.3 卷积层实现
# 10.3.1 自定义权值
x = tf.random.normal([2, 5, 5, 3])  # 模拟输入，3通道，高宽为5
# 需要根据[k, k, cin, cout]格式创建W张量，4个3x3大小卷积核
w = tf.random.normal([3, 3, 3, 4])
# 步长为1，padding为0，
out = tf.nn.conv2d(x, w, strides=1, padding=[[0, 0], [0, 0], [0, 0], [0, 0]])
print(out.shape)

# 步长为1，padding为1，
out = tf.nn.conv2d(x, w, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
print(out.shape)

# 步长为1，padding设置为输出、输入同大小
# 需要注意的是，padding=same只有在strides=1时才是同大小
out = tf.nn.conv2d(x, w, strides=1, padding='SAME')
print(out.shape)

# 高宽先padding成可以整除3的最小整数6，然后6按3倍减少，得到2x2
out = tf.nn.conv2d(x, w, strides=3, padding='SAME')
print(out.shape)

# 根据[cout]格式创建偏置向量
b = tf.zeros([4])
# 在卷积输出上叠加偏置向量，它会自动broadcasting为[b, h', w', cout]
out = out + b

# 10.3.2 卷积层类
layer = layers.Conv2D(4, kernel_size=3, strides=1, padding='SAME')
out = layer(x)  # 前向计算
print(out.shape)
# 返回所有待优化张量列表
print(layer.trainable_variables)
