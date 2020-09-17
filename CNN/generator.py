from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

datagen = ImageDataGenerator(
    rescale=1. / 255,
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

foods_dir = '../images'
food_list = os.listdir(foods_dir)

for food in food_list:
    print(food)
    food_path = foods_dir + '/' + food
    img_paths = os.listdir(food_path)
    if food == './DS_Store':
        continue

    for img in img_paths:
        print(img)
        img_path = food_path + '/' + img
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        save_to_dir = '../gen_images/' + food
        # if not os.path.exists(save_to_dir):
        #     os.mkdir(save_to_dir)
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_prefix='gen_', save_format='jpg'):
            i += 1
            if i > 10:
                break