from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook

import os

foods_dir = "../datasets/train"
# food_list = os.listdir(foods_dir)

f = load_workbook('../datasets/nutrition.xlsx')
xl_sheet = f.active
rows = xl_sheet['F2:F840']
food_list = []
for row in rows:
    for cell in row:
        food_list.append(cell.value)
classes_number = len(food_list)

image_w = 64
image_h = 64
pixels = image_w * image_h * 3

X = []
Y = []
for idx, food in enumerate(food_list):
    label = [0 for _ in range(classes_number)]
    label[idx] = 1

    image_dir = foods_dir + "/" + food
    files = glob.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
    print('{} / {}, {} preprocess complete.'.format(idx, classes_number, food))
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)


np.save("../datasets/dataset.npy", xy)
print('save complete!')
