import tensorflow as tf
from tensorflow.keras import layers

# 9.5.2 L1正则化
# 创建网络参数w1，w2
w1 = tf.random.normal([4, 3])
w2 = tf.random.normal([4, 2])
# 计算L1正则化项
loss_reg = tf.reduce_sum(tf.math.abs(w1)) \
           + tf.reduce_sum(tf.math.abs(w2))
print(loss_reg)

# 9.5.3 L2正则化
# 计算L2正则化项
loss_reg = tf.reduce_sum(tf.square(w1)) \
           + tf.reduce_sum(tf.square(w2))
print(loss_reg)

# 9.6 Dropout
# 添加dropout操作
x = tf.nn.dropout(x, rate=0.5)
# 添加dropout层
model.add(layers.Dropout(rate=0.5))


# 9.7 数据增强
def preprocess(_x, _y):
    # x: 图片的路径，y：图片的数字编码
    _x = tf.io.read_file(_x)
    _x = tf.image.decode_jpeg(_x, channels=3)  # RGBA
    # 图片缩放到244x244大小，这个大小根据网络设定自行调整
    _x = tf.image.resize(_x, [244, 244])


# 9.7.1 旋转
# 图片逆时针旋转180度
x = tf.image.rot90(x, 2)

# 9.7.2 翻转
# 随机水平翻转
x = tf.image.random_flip_left_right(x)
# 随机竖直翻转
x = tf.image.random_flip_up_down(x)

# 9.7.3 裁剪
# 图片先缩放到稍大尺寸
x = tf.image.resize(x, [244, 244])
# 再随机裁剪到合适尺寸
x = tf.image.random_crop(x, [244, 244, 3])
