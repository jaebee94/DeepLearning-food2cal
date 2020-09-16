from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
import pandas as pd

# Image Data Generator
datagen = ImageDataGenerator(
        rotation_range=40,          # 이미지 회전 범위(degrees)
        width_shift_range=0.2,      # 그림을 수평 또는 수직으로 랜덤하게 평행이동 시키는 범위(비율 값)
        height_shift_range=0.2,
        rescale=1./255,             # 0~1 범위로 스케일 변환
        shear_range=0.2,            # 임의 전단 변환 범위
        zoom_range=0.2,             # 임의 확대/축소 범위
        horizontal_flip=True,       # 수평 반전(50% 확률)
        fill_mode='nearest')        # 회전, 이동, 축소할 때 생기는 공간을 채우는 방식

img = load_img('data/train/cats/cat.0.jpg')  # PIL 이미지
x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break

# 엑셀파일 읽기
foods = pd.read_excel('..\\s03p22a411\\datasets\\nutrition.xlsx', usecols=['식품명'])
foods["image_paths"] = '..\\s03p22a411\\datasets\\images' + '\\' + foods["식품명"]
