from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook

# 카테고리 생성
foods_dir = "../images"

### 폴더명으로 카테고리 가져오기 ###
food_list = os.listdir(foods_dir)

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

# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("../data/dataset.npy", allow_pickle=True)

# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256

# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(classes_number))
model.add(Activation('softmax'))

# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])

hdf5_file = "./food_model.hdf5"
model.load_weights(hdf5_file)

test_list = os.listdir('../data/test')
for idx, test in enumerate(test_list):
    test_image = '../data/test/' + test
    img = Image.open(test_image)
    img = img.convert("RGB")
    img = img.resize((64,64))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float") / 256
    X = X.reshape(-1, 64, 64,3)
    # 예측
    pred = model.predict(X)  
    result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
    print('True category : ', test)
    print('New data category : ',food_list[result[0]])