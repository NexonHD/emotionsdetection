import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner.tuners import RandomSearch
import cnn
import config
import utils

MODEL_NAME = 'hyperparameter_sgd'

train_images, train_labels, test_images, test_labels = utils.get_data()

def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Conv2D(hp.Int('conv_filters_1', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_2', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(hp.Int('conv_filters_3', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_4', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(hp.Int('conv_filters_5', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_6', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(hp.Int('conv_filters_7', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_8', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(hp.Int('conv_filters_9', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(hp.Int('conv_filters_10', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(hp.Int('dense_units_1', min_value=256, max_value=2048, step=128), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(hp.Int('dense_units_2', min_value=256, max_value=2048, step=128), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer=cnn.get_custom_optimizer('sgd'), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=3,
    directory=config.KERAS_DIRECTORY,
    project_name='emotion_detector_hyperparameter_tuner')

train_generator, test_generator = cnn.DataGenerators()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

tuner.search(train_generator, epochs=20, validation_data=test_generator, callbacks=[early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

model.summary()

early_stopping_5 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stopping_5])

predictions = model.predict(test_images)

model.save(MODEL_NAME)

_, test_acc = model.evaluate(test_images, test_labels, verbose=0)
utils.print_and_plot_results(history, test_acc)