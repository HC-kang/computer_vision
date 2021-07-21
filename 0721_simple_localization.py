import cv2
import numpy as np
import matplotlib.pyplot as plt
import random;
import tqdm;

IMAGE_SIZE = 200; #200*200
rad = random.randint(5,50); #radius 
c_x = random.randint(rad,IMAGE_SIZE-rad); #x좌표 (random)
c_y = random.randint(rad,IMAGE_SIZE-rad); #y좌표 (random)
blank_image = np.ones(shape=[IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)
cv2.circle(blank_image, (c_x,c_y), rad, 0, -1);
plt.imshow(blank_image);

# 학습 데이터 생성
def create_training_Data():
  l = 10000;  # 10000개 
  X_train = np.zeros(shape=[l,IMAGE_SIZE, IMAGE_SIZE,1]); # shape([10000개, 200, 200, 1 channel])
  Y_train = np.zeros(shape = [l,3]); # x, y 중심좌표, radius >> 3 
  for i in range(l):
    rad = random.randint(5,50);
    c_x = random.randint(rad,IMAGE_SIZE-rad);
    c_y = random.randint(rad,IMAGE_SIZE-rad);
    # normalization 
    Y_train[i,0]= c_x/IMAGE_SIZE;
    Y_train[i,1] = c_y/IMAGE_SIZE;
    Y_train[i,2] = rad/IMAGE_SIZE;
    blank_image = np.ones(shape=[IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8);
    X_train[i,:,:,0] = cv2.circle(blank_image, (c_x,c_y), rad, 0, -1);
  return {'X_Train' : X_train, 'Y_Train': Y_train};
  
training_Data = create_training_Data();

plt.imshow(training_Data['X_Train'][1999].reshape(200,200))
IMAGE_SIZE*training_Data['Y_Train'][1999]

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import Model

img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

x = layers.Conv2D(5, 3, activation='relu', strides = 1, padding = 'same')(img_input) #3*3 kernel 사용 
x = layers.MaxPool2D(pool_size=2)(x)
x = layers.Conv2D(10, 3, activation='relu', strides = 1)(x)
x = layers.MaxPool2D(pool_size=2)(x)
x = layers.Conv2D(20, 3, activation='relu', strides = 1)(x)
x = layers.MaxPool2D(pool_size=2)(x)
x = layers.Conv2D(3, 5, activation='relu', strides = 1)(x)

# TODO
x = layers.Flatten()(x)
output = layers.Dense( 3, activation='relu')(x) #3 >> x좌표, y좌표, radius

model = Model(img_input, output)

model.summary()

model.compile(loss='mean_squared_error',optimizer= 'adam', metrics=['mse']);

model.fit(training_Data["X_Train"],training_Data["Y_Train"], epochs = 3,verbose=1)

# 3번째 이미지에 대해 projection 
IMAGE_SIZE*model.predict(training_Data['X_Train'][3].reshape(1,IMAGE_SIZE, IMAGE_SIZE,1)) 
# reshape(batch_size, image_size, image_size, grey_scale)

# 정답과 비교 
IMAGE_SIZE*training_Data['Y_Train'][3]

plt.imshow(training_Data['X_Train'][3].reshape(200,200))

IMAGE_SIZE = 200;
rad = random.randint(5,50);
c_x = random.randint(rad,IMAGE_SIZE-rad);
c_y = random.randint(rad,IMAGE_SIZE-rad);
print([c_x, c_y, rad])
blank_image = np.ones(shape=[IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)
new_Image = cv2.circle(blank_image, (c_x,c_y), rad, 0, -1);
plt.imshow(new_Image);
print(IMAGE_SIZE*model.predict(new_Image.reshape(1,IMAGE_SIZE, IMAGE_SIZE,1)))