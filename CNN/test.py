from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

import os

foods_dir = "./datasets/train"
foodnames = os.listdir(foods_dir)
classes_number = len(foodnames)

image_w = 64
image_h = 64
pixels = image_w * image_h * 3

X = []
Y = []
for idx, foodname in enumerate(foodnames):
    label = [0 for _ in range(classes_number)]
    label[idx] = 1

    image_dir = foods_dir + "/" + foodname
    files = glob.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
    print('{} / {}, {} preprocess complete.'.format(idx, classes_number, foodname))
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)


np.save("./datasets/dataset.npy", xy)
print('save complete!')
