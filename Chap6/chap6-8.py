import tensorflow as tf
from tensorflow.keras import layers, losses
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 6.8.1 dataset
dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases"
                                                        "/auto-mpg/auto-mpg.data")
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()

# check part of the data
print(dataset.head())
print(dataset.tail())
print(dataset)

# count the blanks in data and clear
print(dataset.isna().sum())
dataset = dataset.dropna()  # delete blanks
print(dataset.isna().sum())
print(dataset)

# deal with types in data, the column origin 1, 2, 3 represent production area: USA, Europe, and Japan
# eject (delete) the column origin
origin = dataset.pop('Origin')
# write in three new columns based on origin
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]], diag_kind="kde")
# plt.show()
# check training input x
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# move MPG as true labels
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# normalize data
def norm(_x):
    return (_x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print(normed_train_data.shape, train_labels.shape)
print(normed_test_data.shape, test_labels.shape)

train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
train_db = train_db.shuffle(100).batch(32)


# 6.8.2 create network
class Network(tf.keras.Model):
    # regression network
    def __init__(self):
        super(Network, self).__init__()
        # create 3 fully connected layers
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs):
        _x = self.fc1(inputs)
        _x = self.fc2(_x)
        _x = self.fc3(_x)

        return _x


# 6.8.3 training and testing
model = Network()
# create inner tensor via build func., None is the batch number, 9 is the length of features
model.build(input_shape=(None, 9))
print(model.summary())
optimizer = tf.keras.optimizers.RMSprop(0.001)  # create the optimizer and specify the learning rate

train_mae_losses = []
test_mae_losses = []
for epoch in range(200):
    for step, (x, y) in enumerate(train_db):  # iterate training set
        # gradient track
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(losses.MSE(y, out))
            mae_loss = tf.reduce_mean(losses.MAE(y, out))

        if step % 10 == 0:
            print(epoch, step, float(loss))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_mae_losses.append(float(mae_loss))
    out = model(tf.constant(normed_test_data.values))
    test_mae_losses.append(tf.reduce_mean(losses.MAE(test_labels, out)))

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(train_mae_losses, label='Train')
plt.plot(test_mae_losses, label='Test')
plt.legend()
plt.show()
