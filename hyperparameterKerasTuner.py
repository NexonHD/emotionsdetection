import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping
import cnn
import config
import utils

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

def get_tuner() -> RandomSearch:
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=3,
        directory=config.KERAS_DIRECTORY,
        project_name='emotion_detector_hyperparameter_tuner')
    return tuner

def early_stopping(patience: int) -> EarlyStopping:
    return EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

def start_tuner_and_get_model_with_best_hps(train_gen, test_gen):
    tuner = get_tuner()

    tuner.search(train_gen, epochs=20, validation_data=test_gen, callbacks=[early_stopping(3)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return tuner.hypermodel.build(best_hps)

def main():
    train_generator, test_generator = cnn.get_datagenerators()

    # get best model of trial
    model = start_tuner_and_get_model_with_best_hps(train_generator, test_generator)

    # print model summary
    model.summary()

    # train model for another 50 epochs
    history = model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stopping(5)])

    # save model
    model.save(config.HP_MODEL_NAME)

    # print results
    _, _, test_images, test_labels = utils.get_data()
    _, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    utils.print_and_plot_results(history, test_acc)

if __name__ == '__main__':
    main()