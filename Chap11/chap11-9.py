# 11.9 LSTM层使用方法
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 11.9.1 LSTMCell
x = tf.random.normal([2, 80, 100])
xt = x[:, 0, :]     # 得到一个时间戳的输入
cell = layers.LSTMCell(64)  # 创建Cell
# 初始化状态和输出List, [h, c]
state = [tf.zeros([2, 64]), tf.zeros([2, 64])]
out, state = cell(xt, state)    # 前向计算
print(id(out), id(state[0]), id(state[1]))

for xt in tf.unstack(x, axis=1):
    out, state = cell(xt, state)

# 11.9.2 LSTM层
layer = layers.LSTM(64)
out = layer(x)

layer = layers.LSTM(64, return_sequences=True)
out = layer(x)

net = Sequential([
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(64)
])
out = net(x)
