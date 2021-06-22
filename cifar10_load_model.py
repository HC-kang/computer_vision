from __future__ import print_function
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import flatten
import numpy as np
import pandas as pd

import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/computer_vision')

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255

input_shape = img_rows, img_cols, 1

print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)

# 모델 바로 불러오기
model = tensorflow.keras.models.load_model('mnist_cnn.h5')
model.summary()

n = 0
plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation = 'nearest')
plt.show()

print('The answer is', np.argmax(model.predict(X_test[n].reshape((1,28,28,1))), axis=-1))

predicted_result = model.predict(X_test)
predicted_labels = np.argmax(predicted_result, axis=1)
test_labels = np.argmax(y_test, axis=1)

wrong_result = []

for n in range(0, len(test_labels)): 
  if predicted_labels[n] != test_labels[n]:
    wrong_result.append(n)

count = 0 
nrows = 4
ncols = 4

plt.figure(figsize=(12,8))

for n in wrong_result:
  count += 1
  plt.subplot(nrows, ncols, count)
  plt.imshow(X_test[n].reshape(28,28), cmap='Greys', interpolation='nearest')
  tmp = 'Label:' + str(test_labels[n]) + ", predictions:" + str(predicted_labels[n])
  plt.title(tmp)

  if count == 16:
    break 

plt.tight_layout()
plt.show()
