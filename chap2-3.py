# linear model
import numpy as np
import matplotlib.pyplot as plt

# 1. sampling data
data = []  # list for saving data
for i in range(100):  # loop sampling 100 points
    x = np.random.uniform(-10., 10.)  # randomly sampling input x
    # sampling Gaussian noise
    eps = np.random.normal(0., 0.1)
    # model output
    y = 1.477 * x + 0.089 + eps
    data.append([x, y])  # save sampling point
data = np.array(data)  # convert to 2D Numpy array


# 2. calculate the error
def mse(b, w, points):
    # calculate the mean squared error based on current parameters w and b
    total_error = 0
    for _i in range(0, len(points)):  # iterate all points
        _x = points[_i, 0]  # i-th input x
        _y = points[_i, 1]  # i-th output y
        # calculate the error squared, and sum
        total_error += (_y - (w * _x + b)) ** 2
    # calculate the mean of the sum of squared error
    return total_error / float(len(points))


# 3. calculate the gradient
def step_gradient(b_current, w_current, points, lr):
    # calculate the derivative of the error function for all points
    # and update w and b
    b_gradient = 0
    w_gradient = 0
    m = float(len(points))  # total number of sampling
    for _i in range(0, len(points)):
        _x = points[_i, 0]
        _y = points[_i, 1]
        # the derivative of error function regarding b:
        # grad_b = w(wx+b-y)
        b_gradient += (2 / m) * ((w_current * _x + b_current) - _y)
        # the derivative of error function regarding w:
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2 / m) * _x * ((w_current * _x + b_current) - _y)
    # update w', b' according to the gradient descent algorithm
    # lr is the learning rate
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


# 4. update the gradient
def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    # iterate to update w and b
    b = starting_b  # the initial value of b
    w = starting_w  # the initial value of w
    losses = []     # list for save losses
    # update based on gradient descent algorithm
    for step in range(num_iterations):
        # calculate the gradient and update once
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)  # calculate the current MSE for controlling the training process
        losses.append(loss)
        if step % 50 == 0:  # print the error and the current values of w and b
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w], losses  # return the last w and b, and the process of loss


def main():
    # load the training data, which is obtained from sampling the real model by adding observation noises
    lr = 0.01  # learning rate
    initial_b = 0  # initialize b
    initial_w = 0  # initialize w
    num_iterations = 1000
    # train and optimize for 1000 times
    # return the optimum w*, b* and the process of training Loss
    [b, w], losses = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)  # calculate the MSE of the optimum w, b
    print(f'Final loss:{loss}, w:{w}, b:{b}')
    plt.plot(losses)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['MSE'])
    plt.show()


if __name__ == "__main__":
    main()
