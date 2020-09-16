import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

len_category = 2262

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len_category))
model.add(Activation('softmax'))

model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 경로 = '../s03p22a411/datasets/images'
train_generator = train_datagen.flow_from_directory(
        '../datasets/train',
        target_size=(150, 150),
        batch_size=50,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        '../datasets/validation',
        target_size=(150, 150),
        batch_size=50,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=len_category,
        validation_data=validation_generator,
        epochs=20)

# model_json = model.to_json()
# with open("model.json", "w") as json_file : 
#     json_file.write(model_json)

# model.save_weights('test_model.h5')

model.save('test_model.h5')
