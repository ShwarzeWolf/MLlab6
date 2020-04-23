import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from sklearn import metrics


(x_train, y_train), (x_test, y_test) = mnist.load_data()

pictures = x_test
targets = y_test

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.0
x_test /= 255.0

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train = model.fit(x=x_train,y=y_train_one_hot, epochs=1)
test_loss, test_acc = model.evaluate(x_test, y_test_one_hot)

print('Точность на проверочных данных:', test_acc)
print('Функция потерь:', test_loss)

predictions = model.predict(x_test)

predictions_classes = []
max_values = [[0.0] * 10 for i in range(10)]
images = [[0] * 10 for i in range(10)]

for i in range(int(predictions.size / 10)):

    predicted_class = np.argmax(predictions[i])
    actual_class = targets[i]

    for j in range(10):
        if predictions[i][j] > max_values[actual_class][j]:
            max_values[actual_class][j] = predictions[i][j]
            images[actual_class][j] = i


    predictions_classes.append(np.argmax(predictions[i]))

confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=predictions_classes)
print(confusion_matrix)
print()
print(max_values)
print(images)

for image_index_row in images:
    for image_index in image_index_row:
        plt.imshow(pictures[image_index], cmap='Greys')
        plt.show()
