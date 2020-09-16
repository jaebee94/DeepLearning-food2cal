from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

foods = pd.read_excel('..\\s03p22a411\\datasets\\nutrition.xlsx', usecols=['식품명'])
foods["image_paths"] = '..\\s03p22a411\\datasets\\images' + '\\' + foods["식품명"]

X, y = foods["image_paths"], foods["식품명"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

train_dataset = pd.merge(X_train, y_train, left_index=True, right_index=True)
val_dataset = pd.merge(X_val, y_val, left_index=True, right_index=True)

with open('.\\datasets\\train_dataset.pickle', 'wb') as file:
    pickle.dump(train_dataset, file)
with open('.\\datasets\\val_dataset.pickle', 'wb') as file:
    pickle.dump(val_dataset, file)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

batch_size = 16

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         'data/train',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')

# validation_generator = validation_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')

# test_generator = test_datagen.flow_from_directory(
#         'data/validation',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')

# model.compile(
#         loss='categorical_crossentropy',
#         optimizer='rmsprop',
#         metrics=['accuracy'])

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=1000//batch_size,
#         validation_data=validation_generator,
#         epochs=50)

# model.save_weights('test_model.h5')
