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

img_rows = 28
img_cols = 28

# 파일 불러오기
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(K.image_data_format())

# shape 변경하기
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

# type 변경해주기
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 정규화
X_train = X_train/255
X_test = X_test/255

# 형태 확인
print( 'X_train shape :', X_train.shape)
print(X_train.shape[0], 'train sample')
print(X_test.shape[0], 'test sample')

# 범주화 - One Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# modeling (출력 0~9)
model = Sequential()
model.add(Conv2D(32, # node 
                 kernel_size=(3,3), # 3X3 kernel
                 activation = 'relu', 
                 input_shape = input_shape
                 )
         )
model.add(Conv2D(64, 
                 (3,3),
                 activation = 'relu',
                 )
         )
model.add(MaxPooling2D(pool_size = (2,2))) # 2x2 구간에서 최대값만 선정 - 특징값 추출하기.
model.add(Dropout(0.25)) # 노드 중 3/4만 활용하기 - 과적합 방지
model.add(Flatten()) # 현재 3차원인 입력값을 1차원으로 변환

# 일종의 전처리 끝. 특징값을을 추출하여 1차원 벡터로 변환하였음.

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

# 출력층.
model.add(Dense(10, activation = 'softmax'))

model.summary()


model.compile(loss = tensorflow.keras.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = 'accuracy')

history = model.fit(X_train, y_train,
                    batch_size = 128,
                    epochs = 4,
                    verbose=1,
                    validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

model.save('mnist_cnn.h5')