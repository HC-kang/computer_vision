%load_ext tensorboard

import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout


num_classes = 2 # 개, 고양이

class Lenet(Sequential):
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(Conv2D(6, kernel_size = (5,5), strides = (1,1), activation = 'relu', 
                        input_shape = input_shape, padding = 'same'))
        
        self.add(AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
        self.add(Conv2D(16, kernel_size = (5,5), strides = (1,1), activation = 'relu', padding = 'valid'))
        self.add(AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
        self.add(Flatten())
        self.add(Dense(120, activation = 'relu'))
        self.add(Dense(84, activation = 'relu'))
        self.add(Dense(nb_classes, activation = 'softmax'))
        
        self.compile(optimizer = 'adam',
                     loss = 'categorical_crossentropy',
                     metrics = 'accuracy')

model = Lenet((100, 100, 3), num_classes)

model.summary()

from google.colab import files
file_uploaded = files.upload()

train_dir = 'catanddog.zip'
!unzip_catanddog.zip

EPOCHS = 100
BATCH_SIZE = 32
image_height = 100
image_width = 100
train_dir = 'train/'
valid_dir = 'validation/'

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

log_dir="./log6-1/"     
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 0)


# 학습
model.fit(train_generator,
          epochs = EPOCHS,
          steps_per_epoch = train_num // BATCH_SIZE,
          validation_data = valid_generator,
          callbacks = [tensorboard_callback],
          verbose = 1)

# verbose 0, 1, 2 - 안함, 훈련과정 막대로, 미니 배치마다 정보 출력

class_names = ['cat', 'dog']
validation, label_batch = next(iter(valid_generator))
prediction_values = model.predict_classes(validation)

fig = plt.figure(figsize = (12, 8))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

for i in range(8):
    ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i, :], cmap = plt.cm.gray_r, interpolation = 'nearest')

    # 올바르게 예측할 때 : green, 틀리게 예측할 때 : red
    if prediction_values[i] == np.argmax(label_batch[i]):
        ax.text(3, 17, class_names[prediction_values[i]], color = 'green', fontsize = 14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color = 'red', fontsize = 14)

    