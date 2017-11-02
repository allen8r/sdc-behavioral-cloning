import csv
import cv2
import numpy as np

rows = []
with open('./data/driving/driving_log.csv') as driving_data:
    reader = csv.reader(driving_data)
    for row in reader:
        rows.append(row)

images = []
steering_angles = []

for row in rows:
    img_path = row[0]
    img_file_name = img_path.split('/')[-1]
    img_path = './data/driving/IMG/'
    image = cv2.imread(img_path + img_file_name)
    images.append(image)

    steering_angle = float(row[3])
    steering_angles.append(steering_angle)

X_train = np.array(images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

model.add(Conv2D(8, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(8, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

