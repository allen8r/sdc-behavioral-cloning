import csv
import cv2
import numpy as np

DATA_PATH = './data/driving/'

rows = []
with open(DATA_PATH + 'driving_log.csv') as driving_data:
    reader = csv.reader(driving_data)
    for row in reader:
        rows.append(row)

images = []
steering_angles = []

for row in rows:
    IMG_PATH = DATA_PATH + 'IMG/'
    for i in range(3):
        img_file_name = row[i].split('/')[-1] # 0:center, 1:left, 2:right
        image = cv2.imread(IMG_PATH + img_file_name)
        images.append(image)
    
    CORRECTION = 0.85
    steering_angle = float(row[3])
    steering_angle_left = steering_angle + CORRECTION
    steering_angle_right = steering_angle - CORRECTION
    steering_angles.extend([steering_angle, steering_angle_left, steering_angle_right])

# Augment data by horizontally flipping the images
augmented_images = []
augmented_steering_angles = []
for image, sterring_angle in zip(images, steering_angles):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image, 1))
    augmented_steering_angles.append(steering_angle)
    augmented_steering_angles.append(steering_angle * -1.0)
    

X_train = np.array(augmented_images)
y_train = np.array(augmented_steering_angles)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Dropout

DROPOUT_RATE = 0.65
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

model.add(Conv2D(8, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(DROPOUT_RATE))
model.add(Conv2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(DROPOUT_RATE))
model.add(Conv2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(DROPOUT_RATE))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(64))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(32))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

model.save('model.h5')