import csv
import cv2
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D


DATA_PATH = './data/'
IMG_PATH = DATA_PATH + 'IMG/'
CAM_CENTER = 0
CAM_LEFT = 1
CAM_RIGHT = 2
DATA_BALANCE = 450


def get_img(cam_position, row_data):
    '''
    Retrieve the camera position image from the provided row data
    '''
    image = cv2.imread(IMG_PATH + row_data[cam_position].split('/')[-1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Process csv data
samples = []
with open(DATA_PATH + 'driving_log.csv') as driving_data:
    reader = csv.reader(driving_data)
    next(reader, None) # skip header row
    
    lines = list(reader)
    random.shuffle(lines)
    
    # Filter out most of the zero angle frames but leave about same amount as the non-zero angle frames
    zero_angle_count = 0
    for line in lines:
        if float(line[3]) == 0.0 and zero_angle_count < DATA_BALANCE:
            samples.append(line)
            zero_angle_count += 1
        elif float(line[3]) != 0.0:
            samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def batch_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            # Extract images and steering angles
            images = []
            steering_angles = []

            for batch_sample in batch_samples:
                images.extend([get_img(CAM_CENTER, batch_sample), get_img(CAM_LEFT, batch_sample), get_img(CAM_RIGHT, batch_sample)])
                
                CORRECTION = 0.65
                steering_angle_center = float(batch_sample[3])
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
    
            X_batch = np.array(augmented_images)
            y_batch = np.array(augmented_steering_angles)
            
            yield sklearn.utils.shuffle(X_batch, y_batch)


# Build the model for training
DROPOUT_RATE = 0.50
BATCH_SIZE = 64
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((65, 25), (0, 0))))

model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation='relu'))
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
# Adam optimizer with default parameters: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
model.compile(optimizer='adam', loss='mse')

train_generator = batch_generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = batch_generator(validation_samples, batch_size=BATCH_SIZE)

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples) // BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples) // BATCH_SIZE,
                    epochs=10)

model.save('model.h5')
