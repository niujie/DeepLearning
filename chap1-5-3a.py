# GPU accelerated compute
import timeit
import matplotlib.pyplot as plt
import tensorflow as tf


def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c


def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c


cpu_time = []
gpu_time = []
n_list = []

for i in range(9):
    n = 10 ** i
    n_list.append(n)
    # create two matrices on CPU
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([1, n])
        cpu_b = tf.random.normal([n, 1])
        print(cpu_a.device, cpu_b.device)

    # create two matrices on GPU
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([1, n])
        gpu_b = tf.random.normal([n, 1])
        print(gpu_a.device, gpu_b.device)

    # warm up to avoid time waste for initialization
    cpu_t = timeit.timeit(cpu_run, number=10)
    gpu_t = timeit.timeit(gpu_run, number=10)
    print('warm up:', cpu_t, gpu_t)

    # formal compute
    cpu_time.append(timeit.timeit(cpu_run, number=10) * 1000)  # * 1000 convert s to ms
    gpu_time.append(timeit.timeit(gpu_run, number=10) * 1000)
    print('run time:', cpu_time[i], gpu_time[i])

plt.loglog(n_list, gpu_time, marker='^')
plt.loglog(n_list, cpu_time, marker='s')
plt.legend(["GPU", "CPU"])
plt.xlabel("Matrix size")
plt.ylabel("Computation Time (ms)")
plt.show()
