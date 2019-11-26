import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf


# optimize Himmelblau func.
def himmelblau(_x):
    return (_x[0] ** 2 + _x[1] - 11) ** 2 + (_x[0] + _x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x, y range: ', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X, Y maps: ', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('3D plot of Himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.figure('Contour plot of Himmelblau')
plt.contour(X, Y, np.log(Z), levels=100)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# initial values
# [1., 0.], [-4, 0.], [4, 0.]
x = tf.constant([-2., 2.])

for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    # back-propagation
    grads = tape.gradient(y, [x])[0]
    # update parameters, 0.01 is the learning rate
    x -= 0.01 * grads
    if step % 20 == 19:
        print('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))
