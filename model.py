import csv
import cv2
import numpy as np
import random


DATA_PATH = './data/'
IMG_PATH = DATA_PATH + 'IMG/'
CAM_CENTER = 0
CAM_LEFT = 1
CAM_RIGHT = 2
DATA_BALANCE = 300


def get_img(cam_position, row_data):
    image = cv2.imread(IMG_PATH + row_data[cam_position].split('/')[-1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Process csv data
rows = []
with open(DATA_PATH + 'driving_log.csv') as driving_data:
    reader = csv.reader(driving_data)
    next(reader, None) # skip header row
    
    lines = list(reader)
    random.shuffle(lines)
    
    # Filter out most of the zero angle frames but leave about same amount as the non-zero angle frames
    zero_angle_count = 0
    for line in lines:
        if float(line[3]) == 0.0 and zero_angle_count < DATA_BALANCE:
            rows.append(line)
            zero_angle_count += 1
        elif float(line[3]) != 0.0:
            rows.append(line)

# Extract images and steering angles
images = []
steering_angles = []

for row in rows:
    images.extend([get_img(CAM_CENTER, row), get_img(CAM_LEFT, row), get_img(CAM_RIGHT, row)])
    
    CORRECTION = 0.75
    steering_angle_center = float(row[3])
    steering_angle_left = steering_angle_center + CORRECTION
    steering_angle_right = steering_angle_center - CORRECTION
    steering_angles.extend([steering_angle_center, steering_angle_left, steering_angle_right])

# Augment the data by horizontally flipping the images
augmented_images = []
augmented_steering_angles = []
for image, steering_angle in zip(images, steering_angles):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image, 1))
    augmented_steering_angles.append(steering_angle)
    augmented_steering_angles.append(steering_angle * -1.0)
    


X_train = np.array(augmented_images)
y_train = np.array(augmented_steering_angles)
print(len(X_train))

# Build the model for training
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D

DROPOUT_RATE = 0.55
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((65, 25), (0, 0))))

model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(64))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(32))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(1))

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')
