import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch

DATA_LOCATION = 'C:/Users/Yannick/Codes/Emotion Detector/dataset/data.npz'
MODEL_NAME = 'hyperparameter_10_2'

mergeddata_dict = np.load(DATA_LOCATION)

train_images = mergeddata_dict['0']
train_labels = mergeddata_dict['1']
test_images = mergeddata_dict['2']
test_labels = mergeddata_dict['3']

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.transpose(train_images, (2, 0, 1))
test_images = np.transpose(test_images, (2, 0, 1))

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Conv2D(hp.Int('conv_filters_1', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_2', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(hp.Int('conv_filters_3', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_4', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(hp.Int('conv_filters_5', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_6', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(hp.Int('conv_filters_7', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_8', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(hp.Int('conv_filters_9', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_10', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())

    model.add(layers.Dense(hp.Int('dense_units_1', min_value=256, max_value=1024, step=128), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(hp.Int('dense_units_2', min_value=256, max_value=1024, step=128), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=3,
    directory='C:/Users/Yannick/Codes/Emotion Detector/kerasHyperparameterTuner',
    project_name='emotion_detector')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

tuner.search(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels), callbacks=[early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

model.summary()

early_stopping_5 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels), callbacks=[early_stopping_5])

predictions = model.predict(test_images)

model.save('C:/Users/Yannick/Codes/Emotion Detector/' + MODEL_NAME)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\n *** Finale Trefferquote auf Testdaten:', test_acc)

plt.plot(history.history['accuracy'], label='Trainingsdaten')
plt.plot(history.history['val_accuracy'], label='Testdaten')
plt.xlabel('Epoche'), plt.ylabel('Trefferquote'), plt.legend(loc='lower right')
plt.show()
