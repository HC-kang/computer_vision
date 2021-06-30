import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.layers.pooling import MaxPool2D

num_classes = 2
class Alexnet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(Conv2D(96, 
                        kernel_size = (11,11), 
                        strides = 4, 
                        padding = 'valid', 
                        activation = 'relu', 
                        input_shape = input_shape, 
                        kernel_initializer = 'he_normal')
                )
        
        self.add(MaxPooling2D(pool_size = (3,3),
                              strides = (2,2),
                              padding = 'valid', 
                              data_format = None)
                )
        # data format default : channel_last(배치크기, 높이, 너비, 채널 수)

        self.add(Conv2D(256,
                        kernel_size = (5,5),
                        strides = 1,
                        padding = 'same',
                        activation = 'relu',
                        kernel_initializer = 'he_normal')
                )

        self.add(MaxPooling2D(pool_size = (3,3), 
                              strides = (2,2),
                              padding = 'valid',
                              data_format = None)
                )

        self.add(Conv2D(384,
                        kernel_size = (3,3),
                        strides = 1,
                        padding = 'same',
                        activation = 'relu',
                        kernel_initializer = 'he_normal')
                )

        self.add(Conv2D(384,
                        kernel_size = (3,3),
                        strides = 1,
                        padding = 'same',
                        activation = 'relu',
                        kernel_initializer = 'he_normal')
                )

        self.add(Conv2D(256,
                        kernel_size = (3,3),
                        strides = 1,
                        padding = 'same',
                        activation = 'relu',
                        kernel_initializer = 'he_normal')
                )

        self.add(MaxPooling2D(pool_size = (3,3), 
                              strides = (2,2),
                              padding = 'valid',
                              data_format = None)
                )

        