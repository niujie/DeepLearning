# 11.1 序列表示方法
# 11.1.1 Embedding层
import tensorflow as tf
from tensorflow.keras import layers

x = tf.range(10)
x = tf.random.shuffle(x)
# 创建共10个单词，每个单词用长度为4的向量表示的层
net = layers.Embedding(10, 4)
out = net(x)
print(out)
print(net.embeddings)
print(net.embeddings.trainable)
