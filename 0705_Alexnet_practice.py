import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten,Dense, Dropout
import matplotlib.pyplot as plt
import os
import time

from tensorflow.python.ops.gen_batch_ops import Batch

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train.shape

num_classes = 10
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train.shape

validation_images, validation_labels = x_train[:500], y_train[:500]

train_images, train_labels = x_train[500:], y_train[500:]

train_images.shape

model = keras.models.Sequential([
    
    Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(32,32,3)),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2 ,2 )),
    Conv2D(256, kernel_size = (5,5), padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size = (3,3), strides = (2,2)),
    Conv2D(384, kernel_size = (3,3) , padding="same"),
    BatchNormalization(),
    Conv2D(384, kernel_size = (1,1), padding="same"),
    BatchNormalization(),
    Conv2D(256, kernel_size = (1,1), padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Flatten(),
    Dense(4096),
    Dropout(0.5),
    Dense(4096),
    Dropout(0.5),
    Dense(10 , activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

train_datagen.fit(train_images)
validation_datagen.fit(validation_images)

history = model.fit_generator(train_datagen.flow(train_images,train_labels, batch_size = 32), 
                    validation_data = validation_datagen.flow(validation_images, validation_labels, batch_size = 32),
                    epochs = 10)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()