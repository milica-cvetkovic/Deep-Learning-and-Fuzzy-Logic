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

output = []

for i in Xtrain.file_paths:
    if "Audi" in i:
        output.append("Audi")
    elif "Hyundai Creta" in i:
        output.append("Hyundai Creta")
    elif "Mahindra Scorpio" in i:
        output.append("Mahindra Scorpio")
    elif "Rolls Royce" in i:
        output.append("Rolls Royce")
    elif "Swift" in i:
        output.append("Swift")
    elif "Tata Safari" in i:
        output.append("Tata Safari")
    elif "Toyota Innova" in i:
        output.append("Toyota Innova")

print("Broj odbiraka Audi klase: ", output.count("Audi"))
print("Broj odbiraka Hyundai Creta klase: ", output.count("Hyundai Creta"))
print("Broj odbiraka Mahindra Scorpio klase: ", output.count("Mahindra Scorpio"))
print("Broj odbiraka Rolls Royce klase: ", output.count("Rolls Royce"))
print("Broj odbiraka Swift klase: ", output.count("Swift"))
print("Broj odbiraka Tata Safari klase: ", output.count("Tata Safari"))
print("Broj odbiraka Toyora Innova klase: ", output.count("Toyota Innova"))

plt.figure()
plt.hist(output)
plt.show()


plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')
plt.show()

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1)
    ]
)

plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
plt.show()

num_classes = len(classes)

model = Sequential(
    [
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(64,64,3)),
        layers.Conv2D(16,3,padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(96, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ]
)

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

history = model.fit(Xtrain,
                    epochs=100,
                    validation_data=Xtest,
                    verbose=0)

model.save("cnn_model")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()


labels = np.array([])
pred = np.array([])
for img, lab in Xtest:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

print('Tacnost modela: ' + str(100*accuracy_score(labels, pred)) + '%.')

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in Xtrain:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

print('Tacnost modela: ' + str(100*accuracy_score(labels, pred)) + '%.')

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()



