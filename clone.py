import csv
import cv2
import numpy as np

images = []
measurements = []

if 1:
    swerve_lines = []
    with open("swerve/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            swerve_lines.append(line)

    for line in swerve_lines:
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'swerve/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 1:
    sinusoid_lines=[]
    with open("sinusoid/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            sinusoid_lines.append(line)

    for line in sinusoid_lines:
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'sinusoid/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 0:
    rec_left_lines=[] #change here
    with open("recover_left/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            rec_left_lines.append(line) #change here

    for line in rec_left_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'recover_left/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 0:
    rec_right_lines=[]
    with open("recover_right/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            rec_right_lines.append(line)

    for line in rec_right_lines:
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'recover_right/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 0:
    rec_right2_lines=[] #change here
    with open("recover_right2/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            rec_right2_lines.append(line) #change here

    for line in rec_right2_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'recover_right2/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 0:
    rec_right3_lines=[] #change here
    with open("recover_right3/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            rec_right3_lines.append(line) #change here

    for line in rec_right3_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'recover_right3/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 0:
    smooth_lines=[] #change here
    with open("smooth/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            smooth_lines.append(line) #change here

    for line in smooth_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'smooth/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 0:
    smoothreverse_lines=[] #change here
    with open("smoothreverse/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            smoothreverse_lines.append(line) #change here

    for line in smoothreverse_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'smoothreverse/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 1:
    sinusoidmtn_lines=[] #change here
    with open("sinusoidmtn/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            sinusoidmtn_lines.append(line) #change here

    for line in sinusoidmtn_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'sinusoidmtn/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 0:
    smoothmtn_lines=[] #change here
    with open("smoothmtn/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            smoothmtn_lines.append(line) #change here

    for line in smoothmtn_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'smoothmtn/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

if 1:
    slalom_lines=[] #change here
    with open("slalom/driving_log.csv") as csvfile: #change here
        reader = csv.reader(csvfile)
        for line in reader:
            slalom_lines.append(line) #change here

    for line in slalom_lines:# change here
        source_path = line[0]
        filename = source_path.split("\\")[-1]
        current_path = 'slalom/IMG/' + filename #change here
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


X_train = np.array(images)
y_train = np.array(measurements)
input_shape = X_train.shape
print(X_train)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers.convolutional import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(6,5,5, subsample=(2,2),border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.3, shuffle = True, nb_epoch = 1)

model.save('model.h5')
