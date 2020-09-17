from keras import optimizers
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
foods_dir = '../gen_images'

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

# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("../data/dataset.npy", allow_pickle=True)

# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float")  / 255
print(X_train.shape[1:])

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
model.add(Dense(256))   # 출력
model.add(Activation('relu'))
# model.add(Dense())   # 출력
# model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(classes_number))
# model.add(Dense(1))
model.add(Activation('softmax'))
# model.add(Activation('sigmoid'))

# 모델 구축하기
adam = optimizers.Adam(lr = 0.001)
model.compile(loss='binary_crossentropy',   # 최적화 함수 지정
    optimizer='adam',
    metrics=['accuracy'])

file_list = os.listdir("../model")
hdf5_file = "../model/" + file_list.pop()
model.load_weights(hdf5_file)

X = []
test_list = os.listdir('../data/test')
if '.DS_Store' in test_list:
    test_list.remove('.DS_Store')
for idx, test in enumerate(test_list):
    print(idx, test)
    test_image = '../data/test/' + test
    img = Image.open(test_image)
    img = img.convert("RGB")
    img = img.resize((150,150))
    data = np.asarray(img)
    X.append(data)
    # X = X.astype("float") / 256
    # X = X.reshape(-1, 64, 64,3)
    # 예측
X = np.array(X)
pred = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(food_list)
# print(pred)
# print('True category : ', test)
print(pred)
cnt = 0
for i in pred:
    pre_ans = i.argmax()
    print(test_list[cnt], food_list[pre_ans])
    cnt += 1