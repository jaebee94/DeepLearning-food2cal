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

# 카테고리 지정
# foods_dir = "../datasets/train"
# foodnames = os.listdir(foods_dir)
# classes_number = len(foodnames)

foods_dir = "../datasets/train"
# food_list = os.listdir(foods_dir)

f = load_workbook('../datasets/nutrition.xlsx')
xl_sheet = f.active
rows = xl_sheet['F2:F100']
food_list = []
for row in rows:
    for cell in row:
        food_list.append(cell.value)
classes_number = len(food_list)

# 이미지 크기 지정하기
# image_w = 64
# image_h = 64
# pixels = image_w * image_h * 3

# X = []
# Y = []
# for idx, food in enumerate(food_list):
#     label = [0 for _ in range(classes_number)]
#     label[idx] = 1

#     image_dir = foods_dir + "/" + food
#     files = glob.glob(image_dir + "/*.jpg")
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.asarray(img)
#         X.append(data)
#         Y.append(label)
#     print('{} / {}, {} preprocess complete.'.format(idx, classes_number, food))
# X = np.array(X)
# Y = np.array(Y)

# X_train, X_test, y_train, y_test = train_test_split(X, Y)
# xy = (X_train, X_test, y_train, y_test)


# np.save("../datasets/dataset.npy", xy)
# print('save complete!')



# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("../datasets/dataset.npy", allow_pickle=True)

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
model.add(Dense(1024))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(classes_number))
model.add(Activation('softmax'))

# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])

# 모델 확인
print(model.summary())

# 학습 완료된 모델 저장
hdf5_file = "./food_model.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(X_train, y_train, batch_size=32, epochs=20)
    model.save_weights(hdf5_file)


# 모델 평가하기 
score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc


# 적용해볼 이미지 
# test_image = '../datasets/test/라면테스트1.jpeg'
# # 이미지 resize
# img = Image.open(test_image)
# img = img.convert("RGB")
# img = img.resize((64,64))
# data = np.asarray(img)
# X = np.array(data)
# X = X.astype("float") / 256
# X = X.reshape(-1, 64, 64,3)
# # 예측
# pred = model.predict(X)  
# result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
# print('New data category : ',food_list[result[0]])
# for i in range(len(result)):
#     print(food_list[result[i]])

test_list = os.listdir('../datasets/test')
for idx, test in enumerate(test_list):
    test_image = '../datasets/test' + test
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