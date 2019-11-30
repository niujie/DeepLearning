# 11.10 GRU简介
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 11.10.3 GRU使用方法
# 初始化状态向量
x = tf.random.normal([2, 80, 100])
h = [tf.zeros([2, 64])]
cell = layers.GRUCell(64)   # 新建GRU Cell
for xt in tf.unstack(x, axis=1):
    out, h = cell(xt, h)
print(out.shape)

net = Sequential([
    layers.GRU(64, return_sequences=True),
    layers.GRU(64)
])
out = net(x)
