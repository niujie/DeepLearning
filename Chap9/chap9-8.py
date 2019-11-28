import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, regularizers
from tensorflow.keras.layers import Dense
import numpy as np

# 9.8 过度拟合问题
# 9.8.1 构建数据集
# 导入数据集生成工具
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 从moon分布中随机采样1000个点，并切分为训练集-测试集
N_SAMPLES = 1000  # 采样点数
TEST_SIZE = 0.3  # 测试数量比率
X, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)


# 绘制数据集的分布，X为2D坐标，y为数据点的标签
def make_plot(_X, _y, plot_name, file_name=None, XX=None, YY=None, preds=None):
    plt.figure()
    # sns.set_style("whitegrid")
    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    # 根据网络输出绘制预测曲面
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.08, cmap=plt.cm.get_cmap("Spectral"))
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.5)
    # 绘制散点图，根据标签区分颜色
    markers = ['o' if i == 1 else 's' for i in _y.ravel()]
    plt.scatter(_X[:, 0], _X[:, 1], c=_y.ravel(), s=20, cmap=plt.cm.get_cmap("Spectral"), edgecolors='none', m=markers)

    if file_name is not None:
        plt.savefig(OUTPUT_DIR + '/' + file_name)
        plt.close()


# 调用make_plot函数绘制数据的分布，其中X为2D坐标，y为标签
make_plot(X, y, None, "dataset.svg")

# 9.8.2 网络层数的影响
for n in range(5):  # 构建5种不同层数的网络
    model = Sequential()  # 创建网络
    # 创建第一层
    model.add(Dense(8, input_dim=2, activation='relu'))
    for _ in range(n):  # 添加n层，共n+2层
        model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 创建最末层
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')  # 模型装配与训练
    history = model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
    # 绘制不同层数的网络决策边界曲线
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "网络层数({})".format(n)
    file = "网络容量%f.png" % (2 + n * 1)
    make_plot(X_train, y_train, title, file, XX, YY, preds)

# 9.8.3 Dropout的影响
for n in range(5):  # 构建5种不同数量Dropout层的网络
    model = Sequential()  # 创建网络
    # 创建第一层
    model.add(Dense(8, input_dim=2, activation='relu'))
    counter = 0
    for _ in range(n):  # 网络层数固定为5
        model.add(Dense(64, activation='relu'))
        if counter < n:  # 添加n个Dropout层
            counter += 1
            model.add(layers.Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid'))  # 输出层
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')  # 模型装配
    # 训练
    history = model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
    # 绘制不同层数的网络决策边界曲线
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "Dropout({})".format(n)
    file = "Dropout%f.png" % (n)
    make_plot(X_train, y_train, title, file, XX, YY, preds)


# 9.8.4 正则化的影响
def build_model_with_regularization(_lambda):
    # 创建带正则化的神经网络
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))  # 不带正则化项
    model.add(Dense(256, activation='relu',  # 带L2正则化项
                    kernel_regularizer=reularizers.l2(_lambda)))
    model.add(Dense(256, activation='relu',  # 带L2正则化项
                    kernel_regularizer=reularizers.l2(_lambda)))
    model.add(Dense(256, activation='relu',  # 带L2正则化项
                    kernel_regularizer=reularizers.l2(_lambda)))
    # 输出层
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')  # 模型装配
    return model


for _lambda in [1e-5, 1e-3, 1e-1, 0.12, 0.13]:  # 设置不同的正则化系数
    # 创建带正则化项的模型
    model = build_model_with_regularization(_lambda)
    # 模型训练
    history = model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
    # 绘制权值范围
    layer_index = 2
    plot_title = "正则化-[lambda = {}]".format(str(_lambda))
    file_name = "正则化_" + str(_lambda)
    # 绘制网络权值范围图
    plot_weights_matirx(model, layer_index, plot_title, file_name)
    # 绘制不同正则化系数的决策边界线
    preds = model.predict_classes(np.c_[XX.avel(), YY.ravel()])
    title = "正则化".format(_lambda)
    file = "正则化%f.svg" % _lambda
    make_plot(X_train, y_train, title, file, XX, YY, preds)
