import datetime
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt


from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers.pooling import AveragePooling2D

# The data, split between train and test sets:
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_train[0].shape, 'image shape')

X_train = X_train[:, :, :, np.newaxis]
X_test = X_test[:, :, :, np.newaxis]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_train[0].shape, 'image shape')

y_test

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

input_shape = X_train[0].shape

model = keras.models.Sequential([
    keras.layers.Conv2D(6, 
                        kernel_size = (5,5), 
                        strides = (1,1), 
                        activation = 'tanh', 
                        input_shape=input_shape, 
                        padding = 'same'),
    keras.layers.AveragePooling2D(pool_size=(2,2),
                                  strides=(2,2),
                                  padding = 'valid'),
    keras.layers.Conv2D(16, 
                        kernel_size = (5,5),
                        strides=(1,1),
                        activation = 'tanh',
                        padding = 'valid'),
    keras.layers.AveragePooling2D(pool_size=(2,2),
                                  strides = (2,2),
                                  padding = 'valid'),
    keras.layers.Conv2D(120, 
                        kernel_size = (5,5), 
                        strides = (1,1),
                        activation = 'tanh',
                        padding = 'valid'),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation = 'tanh'),
    keras.layers.Dense(84, activation = 'tanh'),
    keras.layers.Dense(num_classes, activation = 'softmax')
])

model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['accuracy'])

model.summary()

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(
                    X_train, 
                    y_train,
                    epochs = 5,
                    batch_size = 32,
                    validation_data = (X_test, y_test),
                    callbacks = [tensorboard_callback],
                    verbose = 1)

acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(5)

plt.figure(figsize = (8,8))
plt.subplot(121)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.legend(loc = 'best')
plt.title('Training Accuracy')

plt.subplot(122)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.legend(loc = 'upper right')
plt.title('Training Loss')
plt.show()

test_score = model.evaluate(X_test, y_test)
print('Test loss {:.4f}, accuracy {:.2f}%'.format(test_score[0], test_score[1] * 100))
