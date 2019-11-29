# 11.6 梯度弥散和梯度爆炸
import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.ones([2, 2])     # 任意创建某矩阵
eigenvalues = tf.linalg.eigh(W)[0]  # 计算特征值
print(eigenvalues)

val = [W]
for i in range(10):     # 矩阵相乘n次方
    val.append([val[-1]@W])
# 计算L2范数
norm = list(map(lambda x: tf.norm(x).numpy(), val))
plt.figure()
plt.plot(range(1, 12), norm)
plt.xlabel('n times')
plt.ylabel('L2-norm')
# plt.show()

W = tf.ones([2, 2]) * 0.4   # 任意创建某矩阵
eigenvalues = tf.linalg.eigh(W)[0]  # 计算特征值
print(eigenvalues)
val = [W]
for i in range(10):     # 矩阵相乘n次方
    val.append([val[-1]@W])
# 计算L2范数
norm = list(map(lambda x: tf.norm(x).numpy(), val))
plt.figure()
plt.plot(range(1, 12), norm)
plt.xlabel('n times')
plt.ylabel('L2-norm')
# plt.show()

# 11.6.1 梯度裁剪
a = tf.random.uniform([2, 2])
print(tf.clip_by_value(a, 0.4, 0.6))   # 梯度值裁剪

a = tf.random.uniform([2, 2]) * 5
# 按范数方式裁剪
b = tf.clip_by_norm(a, 5)
print(tf.norm(a), tf.norm(b))

w1 = tf.random.normal([3, 3])   # 创建梯度张量1
w2 = tf.random.normal([3, 3])   # 创建梯度张量2
# 计算global norm
global_norm = tf.math.sqrt(tf.norm(w1)**2+tf.norm(w2)**2)
print(global_norm)
# 根据global norm和max norm=2裁剪
(ww1, ww2), global_norm = tf.clip_by_global_norm([w1, w2], 2)
# 计算裁剪后的张量组的global norm
global_norm2 = tf.math.sqrt(tf.norm(ww1)**2+tf.norm(ww2)**2)
print(global_norm, global_norm2)

plt.show()

'''
with tf.GradientTape() as tape:
    logit = model(x)    # 前向传播
    loss = criteon(y, logits)   # 误差计算
# 计算梯度值
grads = tape.gradient(loss, model.trainable_variables)
grads, _ = tf.clip_by_global_norm(grads, 25)    # 全局梯度裁剪
# 利用裁剪后的梯度张量更新参数
optimizer.apply_gradients(zip(grads, model.trainable_variables))
'''