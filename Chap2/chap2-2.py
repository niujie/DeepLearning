import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-10, 10, 0.01)
y = x ** 2 * np.sin(x)
yp = 2 * x * np.sin(x) + x ** 2 * np.cos(x)
plt.plot(x, y)
plt.plot(x, yp, linestyle='dotted')
plt.legend(["function", "derivative"])
plt.show()

x = np.arange(-1.5, 1.5, 0.01)
y = np.arange(-1.5, 1.5, 0.01)
X, Y = np.meshgrid(x, y)
Z = -(np.cos(X) ** 2 + np.cos(Y) ** 2) ** 2

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(X, Y, Z, ccount=10, rcount=10)
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel("f(x,y)")

x = np.linspace(-1.5, 1.5, 20)
dx = x[1] - x[0]
y = np.linspace(-1.5, 1.5, 20)
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
Z = -(np.cos(X) ** 2 + np.cos(Y) ** 2) ** 2
dZ_dX, dZ_dY = np.gradient(Z, dx, dy)
ax.quiver(X, Y, -4.*np.ones_like(X), dZ_dY, dZ_dX, np.zeros_like(X), length=0.1, normalize=True, color='r')
plt.show()
