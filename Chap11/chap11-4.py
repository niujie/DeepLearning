# 11.4 RNN层使用方法
# 11.4.1 SimpleRNNCell
import tensorflow as tf
from tensorflow.keras import layers

cell = layers.SimpleRNNCell(3)
cell.build(input_shape=(None, 4))
print(cell.trainable_variables)

# 初始化状态向量
h0 = [tf.zeros([4, 64])]
x = tf.random.normal([4, 80, 100])
xt = x[:, 0, :]
# 构建输入特征f=100，序列长度s=80，状态长度=64的Cell
cell = layers.SimpleRNNCell(64)
out, h1 = cell(xt, h0)  # 前向计算
print(out.shape, h1[0].shape)
print(id(out), id(h1[0]))
h = h0
# 在序列长度的维度解开输入，得到xt:[b,f]
for xt in tf.unstack(x, axis=1):
    out, h = cell(xt, h)    # 前向计算
# 最终输出可以聚合每个时间戳上的输出，也可以只取最后时间戳的输出
out = out

# 11.4.2 多层SimpleRNNCell网络
x = tf.random.normal([4, 80, 100])
xt = x[:, 0, :]     # 取第一个时间戳的输入x0
# 构建2个Cell，先Cell0，后Cell1
cell0 = layers.SimpleRNNCell(64)
cell1 = layers.SimpleRNNCell(64)
h0 = [tf.zeros([4, 64])]    # cell0的初始状态向量
h1 = [tf.zeros([4, 64])]    # cell1的初始状态向量
for xt in tf.unstack(x, axis=1):
    # xtw作为输入，输出为out0
    out0, h0 = cell0(xt, h0)
    # 上一个cell的输出out0作为本cell的输入
    out1, h1 = cell1(out0, h1)

# 保存上一层的所有时间戳上面的输出
middle_sequences = []
# 计算第一层的所有时间戳上的输出，并保存
for xt in tf.unstack(x, axis=1):
    out0, h0 = cell0(xt, h0)
    middle_sequences.append(out0)
# 计算第二层的所有时间戳上的输出
# 如果不是末层，需要保存所有时间戳上面的输出
for xt in middle_sequences:
    out1, h1 = cell1(xt, h1)

# 11.4.3 SimpleRNN层
layer = layers.SimpleRNN(64)
x = tf.random.normal([4, 80, 100])
out = layer(x)
print(out.shape)

layer = layers.SimpleRNN(64, return_sequences=True)
out = layer(x)
print(out)

net = tf.keras.Sequential([     # 构建2层RNN网络
    # 除最末层外，都需要返回所有时间戳的输出
    layers.SimpleRNN(64, return_sequences=True),
    layers.SimpleRNN(64)
])
out = net(x)
