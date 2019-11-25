import matplotlib
from matplotlib import pyplot as plt

# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# load MNIST dataset
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x_test:', x_test.shape, 'y_test:', y_test)

train_db = tf.data.Dataset.from_tensor_slices((x, y))

# 5.7.1 shuffle data
train_db = train_db.shuffle(1000)

# 5.7.2 batch training
batchsz = 512
train_db = train_db.batch(batchsz)


# 5.7.3 pre-processing
def preprocess(_x, _y):  # self-defined pre-processing function
    # call this func. would automatically load x, y with shape [b, 28, 28], [b]
    print(_x.shape, _y.shape)
    # normalized to 0 ~ 1
    _x = tf.cast(_x, dtype=tf.float32) / 255.
    _x = tf.reshape(_x, [-1, 28 * 28])  # flatten
    _y = tf.cast(_y, dtype=tf.int32)  # convert to tensor with int32
    _y = tf.one_hot(_y, depth=10)  # one-hot encoding

    return _x, _y


# the pre-processing function implemented in function preprocess
train_db = train_db.map(preprocess)
# iterate 20 epochs
train_db = train_db.repeat(20)
