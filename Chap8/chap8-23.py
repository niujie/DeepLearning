import tensorflow as tf
from tensorflow.keras import layers, Sequential, datasets

# 8.2.1 模型装配
network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(4, 28 * 28))
network.summary()

# 导入优化器，损失函数模块
from tensorflow.keras import optimizers, losses

# 采用Adam优化器，学习率为0。01；采用交叉熵损失函数，包含Softmax
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']  # 设置测量指标为准确率
                )


# 8.2.2 模型训练
def preprocess(_x, _y):
    _x = tf.reshape(_x, [-1])
    return _x, _y


# x: [60k, 28, 28],
# y: [60k]
(x, y), (x_test, y_test) = datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

# x: [0~255] => [0~1.]
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

val_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_db = val_db.shuffle(1000).map(preprocess).batch(128)

# 指定训练集为train_db，验证集为val_db，训练5个epochs，每2个epoch验证一次
# 返回训练信息保存在history中
history = network.fit(train_db, epochs=5, validation_data=val_db, validation_freq=2)
print(history.history)

x, y = next(iter(val_db))
print('predict x:', x.shape)
out = network.predict(x)    # 模型预测
print(out)
network.evaluate(val_db)

# 8.3 模型保存与加载
'''
# 8.3.1 张量方式
# 保存模型参数到文件
network.save_weights('weights.ckpt')
print('saved weights.')
del network     # 删除网络对象
# 重新创建相同的网络结构
network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )
# 从参数文件中读取数据并写入当前网络
network.load_weights('weights.ckpt')
print('loaded weights!')

# 8.3.2 网络方式
# 保存模型结构与模型参数到文件
network.save('model.h5')
print('saved total model.')
del network     # 删除网络对象
# 从文件恢复网络结构与网络参数
network = tf.keras.models.load_model('model.h5')
'''

# 8.3.3 SavedModel方式
# 保存模型结构与模型参数到文件
tf.keras.experimental.export_saved_model(network, 'model-savedmodel')
print('export saved model.')
del network     # 删除网络对象
# 从文件恢复网络结构与网络参数
network = tf.keras.experimental.load_from_saved_model('model-savedmodel')
