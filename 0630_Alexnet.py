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
from tensorflow.python.keras.layers.pooling import MaxPooling2D

from tensorflow.keras.optimizers import Adam

num_classes = 2 # cat & dog
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

        # 3 -> 1차원으로 압축
        self.add(Flatten())

        self.add(Dense(4096, activation = 'relu'))
        self.add(Dense(4096, activation = 'relu'))
        self.add(Dense(1000, activation = 'relu'))
        self.add(Dense(num_classes, activation = 'softmax'))


        self.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),
                     loss = 'categorical_crossentropy',
                     metrics = 'accuracy'
                     )

model = Alexnet((100,100,3), num_classes)

model.summary()

EPOCHS = 100
BATCH_SIZE = 32
image_height = 100
image_width = 100
train_dir = 'train'
valid_dir = 'test'

train = ImageDataGenerator(
                  rescale=1./255,
                  rotation_range=10,
                  width_shift_range=0.1,
                  height_shift_range=0.1,
                  shear_range=0.1,
                  zoom_range=0.1)

train_generator = train.flow_from_directory(train_dir,
                                                    target_size=(image_height, image_width),
                                                    color_mode="rgb",
                                                    batch_size=BATCH_SIZE,
                                                    seed=1,
                                                    shuffle=True,
                                                    class_mode="categorical")

valid = ImageDataGenerator(rescale=1.0/255.0)
valid_generator = valid.flow_from_directory(valid_dir,
                                                    target_size=(image_height, image_width),
                                                    color_mode="rgb",
                                                    batch_size=BATCH_SIZE,
                                                    seed=7,
                                                    shuffle=True,
                                                    class_mode="categorical"
                                                    )
train_num = train_generator.samples
valid_num = valid_generator.samples


# 모델 학습
model.fit(train_generator,
          epochs = EPOCHS,
          steps_per_epoch = train_num // BATCH_SIZE,
          validation_data = valid_generator,
          validation_steps = valid_num // BATCH_SIZE,
          verbose = 1
          )

class_names = ['cat', 'dog']
validation, label_batch = next(iter(valid_generator))
prediction_values = model.predict_classes(validation)

fig = plt.figure(figsize = (12,8))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

for i in range(8):
    ax = fig.add_subplot(2, 4, i+1, xticks = [], yticks = [])
    ax.imshow(validation[i, :], cmap = plt.cm.gray_r, interpolation = 'nearest')

    # 올바른 예측 : 노란색, 틀린 예측 : 빨간색
    if prediction_values[i] == np.argmax(label_batch[i]):
        ax.text(3, 17, class_names[prediction_values[i]], color = 'yellow', fontsize = 14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color = 'red', fontsize = 14)
