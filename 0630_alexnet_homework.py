import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import time

(x_train, y_train), (x_test, y_test)  = keras.datasets.cifar10.load_data()

y_train.shape

num_classes = 10
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train.shape

validation_images, validation_labels = x_train[:500], y_train[:500]
train_images, train_labels = x_train[500:], y_train[500:]

train_images.shape

model = keras.models.Sequential([
                                 
                                 
    keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(32,32,3)),
    # 30,30,96
    keras.layers.BatchNormalization(),
    # 30,30,96
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(4 ,4 )),
    # 7,7,96
    keras.layers.Conv2D(256, kernel_size = (5,5), padding="same"),
    # 7,7,256
    keras.layers.BatchNormalization(),
    # 7,7,256
    keras.layers.MaxPool2D(pool_size = (1,1), strides = (3,3)), #TODO:  1. 3,2   2. 2,2   3. 1,3
    # 3,3,256
    keras.layers.Conv2D(384, kernel_size = (3,3) , padding="same"),
    # 3,3,384
    keras.layers.BatchNormalization(),
    # 3,3,384
    keras.layers.Conv2D(384, kernel_size = (1,1), padding="same"),
    # 3,3,384
    keras.layers.BatchNormalization(),
    # 3,3,384
    keras.layers.Conv2D(256, kernel_size = (1,1), padding="same"),
    # 3,3,256
    keras.layers.BatchNormalization(),
    # 3,3,256
    keras.layers.MaxPool2D(pool_size = (2,2)), #TODO:
    # 1,1,256
    keras.layers.Flatten(),
    # 256
    keras.layers.Dense(4096),
    # 4096
    keras.layers.Dropout(0.5),
    # 4096
    keras.layers.Dense(4096),
    # 4096
    keras.layers.Dropout(0.5),
    # 4096
    keras.layers.Dense(10 , activation='softmax')
    # 10
])
model.summary()