%load_ext tensorboard

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


class VGG19(Sequential):
    def __init__(self, input_shape):
        super().__init__()

        self.add(Conv2D(64, 
                        kernel_size=(3,3), 
                        padding = 'same',
                        activation = 'relu',
                        input_shape = input_shape))
        self.add(Conv2D(64,
                        kernel_size = (3,3),
                        activation = 'relu',
                        padding = 'same'))
        self.add(MaxPooling2D(pool_size = (2,2), 
                              strides=(2,2)))


        self.add(Conv2D(128, 
                        kernel_size=(3,3), 
                        padding = 'same',
                        activation = 'relu'))
        self.add(Conv2D(128,
                        kernel_size = (3,3),
                        activation = 'relu',
                        padding = 'same'))
        self.add(MaxPooling2D(pool_size = (2,2), 
                              strides=(2,2)))


        self.add(Conv2D(256, 
                        kernel_size=(3,3), 
                        padding = 'same',
                        activation = 'relu'))
        self.add(Conv2D(256,
                        kernel_size = (3,3),
                        activation = 'relu',
                        padding = 'same'))
        self.add(MaxPooling2D(pool_size = (2,2), 
                              strides=(2,2)))


        self.add(Conv2D(512, 
                        kernel_size=(3,3), 
                        padding = 'same',
                        activation = 'relu'))
        self.add(Conv2D(512,
                        kernel_size = (3,3),
                        activation = 'relu',
                        padding = 'same'))
        self.add(Conv2D(512, 
                        kernel_size=(3,3), 
                        padding = 'same',
                        activation = 'relu'))
        self.add(Conv2D(512,
                        kernel_size = (3,3),
                        activation = 'relu',
                        padding = 'same'))
        self.add(MaxPooling2D(pool_size = (2,2), 
                              strides=(2,2)))


        self.add(Flatten())


        self.add(Dense(4096, activation = 'relu'))
        self.add(Dropout(0.5))

        self.add(Dense(4096, activation = 'relu'))
        self.add(Dropout(0.5))

        self.add(Dense(1000, activation = 'softmax'))

        self.compile(optimizer = tf.keras.optimizers.Adam(0.003),
                     loss = 'categorical_crossentropy',
                     metrics = 'accuracy')


model = VGG19((224,224,3))
model.summary()

# from google.colab import files
# file_uploaded = files.upload()

# 검증용 class 3개 적용
classes = {282:'cat', 681:'notebook, notebook computer',970:'alp'}

# 이미지 모델 적용
# from google.colab import files
# file_uploaded = files.upload()

image1 = cv2.imread('labtop.jpg')
image1 = cv2.resize(image1, (224,224))
plt.figure()
plt.imshow(image1)
image1 = image1[np.newaxis, :]
predicted_value = model.predict_classes(image1)
plt.title(classes[predicted_value[0]])


# vgg19_weights_tf_dim_ordering_tf_kernels.h5