import tensorflow as tf
from tensorflow.keras import metrics

# 8.6.1 新建测量器
# 新建平均测量器，适合Loss数据
loss_meter = metrics.Mean()

for step in range(1000):
    # 8.6.2 写入数据
    # 记录采样的数据
    loss = tf.random.normal([1])
    loss_meter.update_state(float(loss))

    # 8.6.3 读取统计信息
    # 打印统计平均loss
    # print(step, 'loss: ', float(loss_meter.result()))

    # 8.6.4 清除
    if step % 100 == 0:
        # 打印统计的平均loss
        print(step, 'loss: ', float(loss_meter.result()))
        loss_meter.reset_states()  # 清零测量器

'''
# 8.6.5 准确率统计实战
acc_meter = metrics.Accuracy()  # 创建准确率测量器
        # [b, 784] => [b, 10]，网络输出值
        out = network(x)
        # [b, 10] => [b]，经过argmax后计算预测值
        pred = tf.argmax(out, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        # 根据预测值与真实值写入测量器
        acc_meter.update_state(y, pred)
    # 读取统计结果
    print(step, 'Evaluate Acc: ', acc_meter.result().numpy())
    acc_meter.reset_states()    # 清零测量器
'''