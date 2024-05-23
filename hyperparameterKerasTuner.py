import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from kerastuner.tuners import BayesianOptimization
import cnn
import config
from tensorflow.keras.callbacks import ModelCheckpoint
import utils
from tensorflow.keras.callbacks import Callback
import threading


with tf.device('/GPU:0'):

    DATA_LOCATION = 'C:/Users/Yannick/Codes/Emotion Detector/dataset/data.npz'
    MODEL_NAME = 'hyperparameter_sgd_2'

    train_images, train_labels, test_images, test_labels = utils.get_data()

    
    class LivePlotCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(LivePlotCallback, self).__init__()
            self.epochs = []
            self.history = {
                'loss': [],
                'val_loss': [],
                'accuracy': [],
                'val_accuracy': []
            }

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epochs.append(len(self.epochs))
            self.history['loss'].append(logs.get('loss'))
            self.history['val_loss'].append(logs.get('val_loss'))
            self.history['accuracy'].append(logs.get('accuracy'))
            self.history['val_accuracy'].append(logs.get('val_accuracy'))
            
            self.update_plot()

        def update_plot(self):
            plt.clf()

            # Plot the training and validation loss
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.history['loss'], label='Training Loss')
            plt.plot(self.epochs, self.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            # Plot the training and validation accuracy
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.history['accuracy'], label='Training Accuracy')
            plt.plot(self.epochs, self.history['val_accuracy'], label='Validation Accuracy')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            # Pause to update the figure
            plt.pause(0.01)

        def on_train_end(self, logs=None):
            plt.show()

    def build_model(hp):
        """
        model = keras.Sequential()

        model.add(layers.Conv2D(hp.Int('conv_filters_1', min_value=32, max_value=512, step=32), (3, 3), activation='relu', padding='same'))
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

        """

        model = keras.Sequential()
        num_conv_layers = hp.Int('num_conv_layers', 6, 10)
        print(f"Building model with {num_conv_layers} convolutional layers")  # Debugging statement
        
        # Add the first Conv2D layer with input_shape
        conv_filters = hp.Int('conv_filters_0', 64, 512, step=32)
        model.add(layers.Conv2D(filters=conv_filters, kernel_size=(3,3), activation='relu', padding='same', input_shape=(48, 48, 1)))
        model.add(layers.BatchNormalization())
        print(f"Layer 0: conv_filters={conv_filters}")
        
        for i in range(1, num_conv_layers):
            conv_filters = hp.Int('conv_filters_' + str(i), 64, 512, step=32)
            print(f"Layer {i}: conv_filters={conv_filters}")  # Debugging statement
            model.add(layers.Conv2D(filters=conv_filters, kernel_size=(3,3), activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
            if i % 2 == 1:
                model.add(layers.MaxPooling2D((2, 2)))
                model.add(layers.Dropout(0.25))
        
        model.add(layers.Flatten())
        num_dense_layers = hp.Int('num_dense_layers', 1, 3)
        print(f"Building model with {num_dense_layers} dense layers")  # Debugging statement
        for i in range(num_dense_layers):
            dense_units = hp.Int('dense_units_' + str(i), 256, 2048, step=128)
            print(f"Dense Layer {i}: dense_units={dense_units}")  # Debugging statement
            model.add(layers.Dense(units=dense_units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.5))

        model.add(layers.Dense(7, activation='softmax'))

        model.compile(optimizer=cnn.get_custom_optimizer('sgd'), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model
    
    class PrintLearningRateCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = lr(self.model.optimizer.iterations)
            else:
                current_lr = lr
            print(f"\nLearning rate for epoch {epoch + 1} is {current_lr.numpy()}")


    def main():
        tuner = BayesianOptimization(
            build_model,
            objective='val_accuracy',
            max_trials=15,
            executions_per_trial=1,
            directory=config.KERAS_DIRECTORY,
            project_name='emotion_detector_sgd_2')
        


        train_generator, test_generator = cnn.get_datagenerators()

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

        tuner.search(train_generator, epochs=25, validation_data=test_generator, callbacks = [PrintLearningRateCallback(), early_stopping, LivePlotCallback()])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)

        #model.summary()

        #model.save(config.KERAS_DIRECTORY + "untrained_" + MODEL_NAME + ".keras")

        #early_stopping_5 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        checkpoint_path = (config.KERAS_DIRECTORY + MODEL_NAME + ".keras")
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        history = model.fit(train_generator, epochs=200, validation_data=test_generator, callbacks=[checkpoint_callback,LivePlotCallback(),PrintLearningRateCallback()])

        predictions = model.predict(test_images)

        #model.save('C:/Users/Yannick/Codes/Emotion Detector/' + MODEL_NAME)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

        utils.print_and_plot_results(history,test_acc)

    if __name__ == '__main__':
        main()