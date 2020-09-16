# tensorflow, tf.keras 임포트
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 헬퍼(helper) 라이브러리 임포트
import numpy as np
# import matplotlib.pyplot as plt

# 1. 데이터 다운
# test_image = image.load_img('짬뽕테스트.jpeg', target_size=(150, 150))
# test_image = test_image.convert('RGB')
# test_image = test_image / 255.0
# data = np.asarray(test_image)

# 2. 모델 불러오기
# from keras.models import model_from_json
# json_file = open("model.json", "r")
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# loaded_model = loaded_model.load_weights('test_model.h5')
model = keras.models.load_model('test_model.h5')

# 3. 예측 만들기
# predictions = model.predict(data)

# print(predictions)
# # 4. 데이터 시각화
# def plot_image(i, predictions, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

#     plt.imshow(img, cmap=plt.cm.binary)

#     predicted_label = np.argmax(predictions_array)
#     plt.xlabel("Prediction: {} ({} %)\nLabel: {}".format(class_names[predicted_label],
#                                                         round(100*np.max(predictions_array), 2),
#                                                         class_names[true_label]))

# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.grid(False)
#     plot_image(i, predictions, test_labels, test_images)
# plt.tight_layout()
# plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

print("-- Predict --")
output = model.predict_generator(test_generator, steps=1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(test_generator.class_indices)
print(output)