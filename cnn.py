import numpy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

DATA_LOCATION = 'C:/Users/linie/vsc/emotionsdetection/data/data.npz'
MODEL_NAME = '1_0.keras'

mergeddata_dict = numpy.load(DATA_LOCATION)

train_images = mergeddata_dict['0']
train_labels = mergeddata_dict['1']
test_images = mergeddata_dict['2']
test_labels = mergeddata_dict['3']

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = numpy.expand_dims(train_images, -1)
test_images = numpy.expand_dims(test_images, -1)


def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(7, activation='softmax'))

    return model

model = create_model()
#model = tf.keras.models.load_model(MODEL_NAME)


model.compile(optimizer='adam',metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
predictions = model.predict(test_images) # Wendet trainiertes Netz auf alle Test-Bilder an und speichert Vorhersagen ab

model.save(MODEL_NAME)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
print('\n *** Finale Trefferquote auf Testdaten:', test_acc)

plt.plot(history.history['accuracy'], label='Trainingsdaten')
plt.plot(history.history['val_accuracy'], label = 'Testdaten')
plt.xlabel('Epoche'), plt.ylabel('Trefferquote'), plt.legend(loc='lower right')
plt.show()
