import os
import cv2
import numpy as np

img_size = 224
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']

def load_data(data_dir):
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)

                # read image
                img_array = cv2.imread(img_path)

                # resize
                img_array = cv2.resize(img_array, (img_size, img_size))

                # normalize
                img_array = img_array / 255.0

                data.append(img_array)
                labels.append(class_num)

            except:
                pass

    return np.array(data), np.array(labels)
X_train, y_train = load_data("archive/Training")
X_test, y_test = load_data("archive/Testing")
print(X_train.shape)
print(y_train.shape)
