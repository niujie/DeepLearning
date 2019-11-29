# 10.11 卷积层变种
import tensorflow as tf
from tensorflow.keras import layers

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

# 10.11.1 空洞卷积
x = tf.random.normal([1, 7, 7, 1])  # 模拟输入
# 空洞卷积，1个3x3的卷积核
layer = layers.Conv2D(1, kernel_size=3, strides=1, dilation_rate=2)
out = layer(x)
print(out.shape)    # 前向计算

# 10.11.2 转置卷积
# 创建X矩阵，高宽为5x5
x = tf.range(25) + 1
# reshape为合法维度的张量
x = tf.reshape(x, [1, 5, 5, 1])
x = tf.cast(x, tf.float32)
# 创建固定内容的卷积核矩阵
w = tf.constant([[-1, 2, -3.], [4, -5, 6], [-7, 8, -9]])
# 调整为合法维度的张量
w = tf.expand_dims(w, axis=2)
w = tf.expand_dims(w, axis=3)
# 进行普通卷积运算
out = tf.nn.conv2d(x, w, strides=2, padding='VALID')
print(out)
# 普通卷积的输出作为转置卷积的输入，进行转置卷积运算
xx = tf.nn.conv2d_transpose(out, w, strides=2, padding='VALID', output_shape=[1, 5, 5, 1])
print(tf.reshape(xx, [5, 5]))

x = tf.random.normal([1, 6, 6, 1])
# 6x6的输入经过普通卷积
out = tf.nn.conv2d(x, w, strides=2, padding='VALID')
print(out.shape)
# 恢复6x6大小
xx = tf.nn.conv2d_transpose(out, w, strides=2, padding='VALID', output_shape=[1, 6, 6, 1])
print(xx)

# 创建4x4大小的输入
x = tf.range(16) + 1
x = tf.reshape(x, [1, 4, 4, 1])
x = tf.cast(x, tf.float32)
# 创建3x3卷积核
w = tf.constant([[-1, 2, -3.], [4, -5, 6], [-7, 8, -9]])
w = tf.expand_dims(w, axis=2)
w = tf.expand_dims(w, axis=3)
# 普通卷积运算
out = tf.nn.conv2d(x, w, strides=1, padding='VALID')
print(out)
# 恢复4x4大小的输入
xx = tf.nn.conv2d_transpose(out, w, strides=1, padding='VALID', output_shape=[1, 4, 4, 1])
xx = tf.squeeze(xx)
xx = tf.squeeze(xx)
print(xx)

# 创建转置卷积类
layer = layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='VALID')
xx2 = layer(out)
print(xx2)
