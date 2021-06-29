import warnings
warnings.filterwarnings('ignore')

from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# data 분할
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = (32,32,3), activation = 'relu'))
model.add(BatchNormalization())


model.add(Conv2D(32, kernel_size=(3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.2))


model.add(Conv2D(64, kernel_size=(3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.3))


model.add(Conv2D(128, kernel_size=(3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3,3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))


# 10개로 다중 분류
model.add(Dense(10, activation = 'softmax'))


from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr = 0.01)


# 학습 실행 환경설정
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = 'accuracy')

model.summary()


# 모델 훈련
model.fit(train_images, train_labels, epochs = 20, batch_size = 200)


# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc :', test_acc)


# 모델 저장
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/computer_vision')
if not os.path.exists('.model'):
    os.mkdir('model')
