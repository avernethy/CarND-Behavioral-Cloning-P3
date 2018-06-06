import csv
import cv2
import numpy as np
import os
import sklearn
from random import shuffle

#modified version of the example generator here
def generator(samples, batch_size = 256):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                left_path = batch_sample[1]
                right_path = batch_sample[2]
                path_parts = center_path.split("\\")
                center_name = center_path.split("\\")[-1]
                left_name = left_path.split("\\")[-1]
                right_name = right_path.split("\\")[-1]
                image_path_center = path_parts[-3]+'/'+path_parts[-2]+'/'+center_name
                image_path_left = path_parts[-3]+'/'+path_parts[-2]+'/'+left_name
                image_path_right = path_parts[-3]+'/'+path_parts[-2]+'/'+right_name
                center_image = cv2.imread(image_path_center)
                left_image = cv2.imread(image_path_left)
                right_image = cv2.imread(image_path_right)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                #images.append(cv2.flip(center_image,1))
                angles.append(center_angle)
                angles.append(center_angle+0.45)
                angles.append(center_angle-0.45)
                #angles.append(center_angle * -1.0)
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train,y_train)

#intialize the list of images
lines = []

#populate the list of images.  turn on/off the data sets with 1 or 0
if 1:
    with open("ps3_3laps_smooth/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1:
    with open("ps3_3laps_smooth_rev/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1:
    with open("ps3_mtn_smooth_3laps/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1:
    with open("ps3_sooth_mtn_rev/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1:
    with open("recoveryRight4/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1: #more recovery driving
    with open("recoveryLeft2/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader: 
            lines.append(line)
if 0: #Udacity data
    with open("data/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
if 0: #Swerving data
    with open("swerve/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 0: #higher frequency, low amplitude sinusoid
    with open("sinusoid/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1: #recover from the left
    with open("recover_left/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1: #recover from the right
    with open("recover_right/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1: #recover from the right #2
    with open("recover_right2/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1: #recover from the right #3
    with open("recover_right3/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1: #smooth driving
    with open("smooth/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1:  #smooth in reverse
    with open("smoothreverse/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 0: #sinusoid mountain driving
    with open("sinusoidmtn/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 1: #smooth moumntain driving
    with open("smoothmtn/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 0: #more zig zag driving
    with open("slalom/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 0:#edge driving#not good pulls the model to the edge
    with open("right_edge_turn/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if 0:#swerve driving
    with open("swerve3laps/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
if 1:#evevn more smooth driving
    with open("smoothLaps/driving_log.csv") as csvfile: 
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
if 1:#right turn recovery
    with open("rightTurn/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
if 1: #left avoid
    with open("leftAvoid/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

#using the big list of all images sampled, create a training and validation set
from sklearn.model_selection import train_test_split
shuffle(lines)

train_samples, validation_samples = train_test_split(lines, test_size = 0.2)
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers.convolutional import Cropping2D

#NVIDIA model
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(20,20))))#(70,20)
model.add(Convolution2D(24,5,5,subsample=(2,2), border_mode='valid', activation = 'relu'))
#model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))
#model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = "relu"))
#model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
#model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3,validation_data = validation_generator, nb_val_samples = len(validation_samples)*3, nb_epoch = 3, verbose = 1)

#print(history_object.history.keys())
#import matplotlib.pyplot as plt
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
model.save('model.h5')
