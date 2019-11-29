import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 10.8 BatchNorm层
# 10.8.2 反向更新
# 构造输入
x = tf.random.normal([100, 32, 32, 3])
# 将其它维度合并，仅保留通道维度
x = tf.reshape(x, [-1, 3])
# 计算其它维度的均值
ub = tf.reduce_mean(x, axis=0)
print(ub)

# 10.8.3 BN层实现
# 创建BN层
layer = layers.BatchNormalization()
print(layer)

network = Sequential([  # 网络容器
    layers.Conv2D(6, kernel_size=3, strides=1),
    # 插入BN层
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.ReLU(),
    layers.Conv2D(16, kernel_size=3, strides=1),
    # 插入BN层
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    # 此处也可以插入BN层
    layers.Dense(84, activation='relu'),
    # 此处也可以插入BN层
    layers.Dense(10)
])
network.build(input_shape=(4, 28, 28, 3))
network.summary()

'''
    with tf.GradientTape() as tape:
        # 插入通道维度
        x = tf.expand_dims(x, axis=3)
        # 前向计算，设置计算模式，[b, 784] => [b, 10]
        out = network(x, training=True)

        for x, y in db_test:    # 遍历测试集
            # 插入通道维度
            x = tf.expand_dims(x, axis=3)
            # 前向计算，测试模式
            out = network(x, training=False)
'''