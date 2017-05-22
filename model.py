import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense,Cropping2D,Lambda,Activation,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle

path="/home/stambde/Desktop/datset/dataset/"    # dataset path
logfile_obj = open('driving_log.csv')
logfile = csv.reader(logfile_obj)
image_list = []         # list to contain images
steering_angle_list = []    #list to contain steering angle

# Prepare dataset & data augmentation
correction = 0.2
for line in logfile:
    steering_angle = line[3]
    centre_img = line[0]
    left_img = line[1]
    right_img = line[2]

    centre_img_np=cv2.imread(centre_img)                # add centre camera image
    image_list.append(centre_img_np)
    steering_angle_list.append(eval(steering_angle))

    left_img_np=cv2.imread(left_img.strip(' '))         # add left camera image
    image_list.append(left_img_np)
    steering_angle_list.append(eval(steering_angle)+correction)

    right_img_np=cv2.imread(right_img.strip(' '))       # add right camera image
    image_list.append(right_img_np)
    steering_angle_list.append(eval(steering_angle)-correction)    

# Data shuffling
list1_shuf = []
list2_shuf = []
index_shuf = list(range(len(image_list)))
shuffle(index_shuf)
for i in index_shuf:
    list1_shuf.append(image_list[i])
    list2_shuf.append(steering_angle_list[i])

x_train = np.array(image_list)
y_train = np.array(steering_angle_list)

print("X_train:",x_train.shape)
print("y_train:",y_train.shape)


# Creating Keras model
model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))   # image normalization
model.add(Cropping2D(cropping=((70,20),(0,0))))                     # cropping the image
print("new shape:",model.output_shape)

model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
#model.add(Dropout(0.25))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
#model.add(Dropout(0.25))

model.add(Convolution2D(48,3,3,subsample=(2,2),activation='relu'))
#model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

model.add(MaxPooling2D())
#model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,validation_split=0.2,nb_epoch=10,shuffle=True)    
model.save('model.h5')                                      # save the model
