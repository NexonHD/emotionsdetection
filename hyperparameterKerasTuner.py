import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_tuner.tuners import BayesianOptimization, RandomSearch
from keras.callbacks import EarlyStopping
import cnn
import config
from keras.callbacks import ModelCheckpoint
import utils
import matplotlib.pyplot as plt



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

class PrintLearningRateCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = lr(self.model.optimizer.iterations)
            else:
                current_lr = lr
            print(f"\nLearning rate for epoch {epoch + 1} is {current_lr.numpy()}")


def build_model(hp, min_conv_layers = 6, max_conv_layers = 10, min_conv_filters = 64, max_conv_filters = 512, min_dense_layers = 1, max_dense_layers = 3, min_dense_units = 256, max_dense_units = 2048):

    model = keras.Sequential()
    num_conv_layers = hp.Int('num_conv_layers', min_conv_layers, max_conv_layers)
    print(f"Building model with {num_conv_layers} convolutional layers")  # Debugging statement
    
    # Add the first Conv2D layer with input_shape
    conv_filters = hp.Int('conv_filters_0', min_conv_filters, max_conv_filters, step=32)
    model.add(layers.Conv2D(filters=conv_filters, kernel_size=(3,3), activation='relu', padding='same', input_shape=(48, 48, 1)))
    model.add(layers.BatchNormalization())
    print(f"Layer 0: conv_filters={conv_filters}")
    
    # remaining conv2d layers
    for i in range(1, num_conv_layers):
        conv_filters = hp.Int('conv_filters_' + str(i), min_conv_filters, max_conv_filters, step=32)
        print(f"Layer {i}: conv_filters={conv_filters}")  # Debugging statement
        model.add(layers.Conv2D(filters=conv_filters, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        if i % 2 == 1:
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())

    #dense layers
    num_dense_layers = hp.Int('num_dense_layers', min_dense_layers, max_dense_layers)
    print(f"Building model with {num_dense_layers} dense layers")  # Debugging statement
    for i in range(num_dense_layers):
        dense_units = hp.Int('dense_units_' + str(i), min_dense_units, max_dense_units, step=128)
        print(f"Dense Layer {i}: dense_units={dense_units}")  # Debugging statement
        model.add(layers.Dense(units=dense_units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer=cnn.get_custom_optimizer('sgd'), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model
    
def get_tuner() -> BayesianOptimization:
    tuner = BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=1,
        directory=config.KERAS_DIRECTORY,
        project_name='emotion_detector_hptuner_sgd_2')
    return tuner


def early_stopping(patience: int) -> EarlyStopping:
    return EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

def start_tuner_and_get_model_with_best_hps(train_gen, test_gen, epochs = 25):
    tuner = get_tuner()

    tuner.search(train_gen, epochs=epochs, validation_data=test_gen, callbacks=[early_stopping(3), PrintLearningRateCallback(), LivePlotCallback()])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return tuner.hypermodel.build(best_hps)

def main():
    train_generator, test_generator = cnn.get_datagenerators()

    # get best model of trial
    model = start_tuner_and_get_model_with_best_hps(train_generator, test_generator)

    # print model summary
    model.summary()

    # train model for another 50 epochs
    checkpoint_path = (config.KERAS_DIRECTORY + config.HP_MODEL_NAME)
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stopping(5),checkpoint_callback, PrintLearningRateCallback(), LivePlotCallback()])



    # print results
    _, _, test_images, test_labels = utils.get_data()
    _, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    utils.print_and_plot_results(history, test_acc)

if __name__ == '__main__':
    main()