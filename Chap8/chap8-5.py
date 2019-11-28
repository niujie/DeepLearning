import tensorflow as tf
from tensorflow.keras import layers, Sequential

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# 8.5.1 加载模型
# 加载ImageNet预训练网络模型，并去掉最后一层
resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
resnet.summary()
# 测试网络的输出
x = tf.random.normal([4, 224, 224, 3])
out = resnet(x)
print(out.shape)

# 新建池化层
global_average_layer = layers.GlobalAveragePooling2D()
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4, 7, 7, 2048])
out = global_average_layer(x)  # 池化层降维
print(out.shape)

# 新建全连接层
fc = layers.Dense(100)
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4, 2048])
out = fc(x)
print(out.shape)

# 重新包裹成我们的网络模型
mynet = Sequential([resnet, global_average_layer, fc])
mynet.summary()

# 冻结ResNet部分网络参数，只训练新建的网络层，从而快速、高效完成网络模型的训练
resnet.trainable = False
mynet.summary()
