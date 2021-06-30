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
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

y_train.shape

validation_images, validation_labels = x_train[:500], y_train[:500]
train_images, train_labels = x_train[500:], y_train[500:]

train_images.shape

model = keras.models.Sequential([
                                 
                                 
    keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(32,32,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(4 ,4 )),
    keras.layers.Conv2D(  , padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(  ),
    keras.layers.Conv2D(  , padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(  , padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(  , padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D( ),
    keras.layers.Flatten(),
    keras.layers.Dense(  ),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(  ),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(  , activation='softmax')
])