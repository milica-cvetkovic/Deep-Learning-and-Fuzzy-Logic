import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.utils import image_dataset_from_directory
from keras import layers
from keras import Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from tensorflow import keras
import cv

main_path_train = 'C:\\Users\\milic\\OneDrive\\Desktop\\carsdataset\\train'
main_path_test = 'C:\\Users\\milic\\OneDrive\\Desktop\\carsdataset\\test'

img_size = (64, 64)
batch_size = 64

Xtrain = image_dataset_from_directory(
    main_path_train,
    image_size=img_size,
    batch_size=batch_size,
    seed=123
)

Xtest = image_dataset_from_directory(
    main_path_test,
    image_size=img_size,
    batch_size=batch_size,
    seed=123
)

classes = Xtrain.class_names

N = 10

model = keras.models.load_model('C:\\Users\\milic\\PycharmProjects\\cnn\\cnn_model')

img = cv2.imread('rolls.jpg')

plt.imshow(img, cmap=plt.cm.binary)

pred = model.predict(np.array([img]))

pred1 = np.argmax(pred)
print(f'Predikcija je {classes[pred1]}, ispravno: Rolls Royce')

img = cv2.imread('audi1.png')

plt.imshow(img, cmap=plt.cm.binary)

pred = model.predict(np.array([img]))

pred1 = np.argmax(pred)
print(f'Predikcija je {classes[pred1]}, ispravno: Audi')