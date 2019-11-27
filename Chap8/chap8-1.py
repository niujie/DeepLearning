# 8.1.1 常见网络层类
import tensorflow as tf
# 导入keras模型，不能使用import keras，它导入的是标准的Keras库
from tensorflow import keras
from tensorflow.keras import layers     # 导入常见网络层类

x = tf.constant([2., 1., 0.1])
layer = layers.Softmax(axis=-1)     # 创建Softmax层
print(layer(x))    # 调用softmax前向计算

# 8.1.2 网络容器
# 导入Sequential容器
from tensorflow.keras import Sequential
network = Sequential([      # 封装为一个网络
    layers.Dense(3, activation=None),   # 全连接层
    layers.ReLU(),  # 激活函数层
    layers.Dense(2, activation=None),   # 全连接层
    layers.ReLU()   # 激活函数层
])
x = tf.random.normal([4, 3])
print(network(x))   # 输入从第一层开始，逐层传播至最末层

layers_num = 2  # 堆叠2次
network = Sequential([])    # 先创建空的网络
for _ in range(layers_num):
    network.add(layers.Dense(3))    # 添加全连接层
    network.add(layers.ReLU())      # 添加激活函数层

network.build(input_shape=(None, 4))     # 创建网络参数
network.summary()
# 打印网络的待优化参数名与shape
for p in network.trainable_variables:
    print(p.name, p.shape)
