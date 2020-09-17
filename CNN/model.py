from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook

# 카테고리 생성
foods_dir = "../images"
# foods_dir = '../gen_images'

### 폴더명으로 카테고리 가져오기 ###
food_list = os.listdir(foods_dir)
if '.DS_Store' in food_list:
    food_list.remove('.DS_Store')
### 엑셀에서 카테고리 가져오기 ###
# f = load_workbook('../datasets/nutrition.xlsx')
# xl_sheet = f.active
# rows = xl_sheet['F2:F840']
# food_list = []
# for row in rows:
#     for cell in row:
#         food_list.append(cell.value)
############################

classes_number = len(food_list)

print('category : ', food_list, classes_number)
# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("../data/dataset.npy", allow_pickle=True)

# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0
print(X_train.shape, X_train.dtype)
# y_train = np_utils.to_categorical(y_train, classes_number)
# y_test = np_utils.to_categorical(y_test, classes_number)

# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(128))   # 출력
model.add(Activation('relu'))
model.add(Dense(64))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(classes_number))
# model.add(Dense(1))
model.add(Activation('softmax'))
# model.add(Activation('sigmoid'))

# 모델 구축하기
# adam = optimizers.Adam(lr = 0.001)
model.compile(loss='binary_crossentropy',   # 최적화 함수 지정
    optimizer='adam',
    metrics=['accuracy'])

# 모델 확인
print(model.summary())

# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# datagen.fit(X_train)

# # model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
# model.fit(datagen.flow(X_train, y_train, batch_size=30),
#         steps_per_epoch=len(X_train) / 30, epochs=100)
MODEL_SAVE_FOLDER_PATH = '../model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                             verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto')

model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_test, y_test), callbacks=[checkpoint])


# 학습 완료된 모델 저장
hdf5_file = "./food_model.hdf5"
model.save_weights(hdf5_file)


# 모델 평가하기 
score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc


