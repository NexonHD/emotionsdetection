import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import config
import utils
from os.path import isfile

def build_and_compile_model(checkpoint_path):
    # load model, or create if not existent
    try:
        model = tf.keras.models.load_model(config.KERAS_DIRECTORY + config.LOAD_MODEL_FROM_FILE)
    except Exception:
        model = config.create_model()

    # optimizer = get_custom_optimizer('adam')
    # model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # load checkpoint, skip if not existent
    try:
        model.load_weights(checkpoint_path)
    except ValueError and FileNotFoundError:
        print(f'weights were not found at {checkpoint_path = }')
    return model

def get_lr_schedule(type: str = 'exponential', initial_learning_rate = 0.01, end_lr = 0.00001, decay_change_steps = config.EPOCHS, batch_size = 32):
    decay_rate = (float(end_lr/initial_learning_rate))**(1.0/decay_change_steps)
    match type:
        case 'exponential':
            exponential_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=(898 * 32) / batch_size,  # Anpassung der decay_steps für  langsamere Reduktion
                decay_rate=decay_rate,
                staircase=True
            )
            return exponential_lr_schedule
        
        case 'cosine':
            cosine_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=898*config.EPOCHS,
                alpha=end_lr
            )
            return cosine_lr_schedule
        
        case 'constant':
            return initial_learning_rate


def get_custom_optimizer(optimizer: str = 'adam', batch_size = 32):
    lr_schedule = get_lr_schedule(batch_size=batch_size)
    match optimizer:
        case 'adam':
            return keras.optimizers.Adam(learning_rate=lr_schedule)
        case 'sgd':
            return keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

def get_datagenerators(batch_size = 32):
    with tf.device('/CPU:0'):
        train_images, train_labels, test_images, test_labels = utils.get_data()

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator()

        train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)
        test_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)

        return train_generator, test_generator

def check_if_checkpoint_already_exists(checkpoint_path: str):
    if isfile(checkpoint_path):
        print(f'\nCheckpoint: {checkpoint_path} already exists, do you wish to overwrite it?')
        user_input = input('Y/N\n')
        if user_input == 'N':
            quit()
        if not user_input == 'Y':
            print(f'invalid input: {user_input}')
            check_if_checkpoint_already_exists(checkpoint_path)

def train():
    checkpoint_path = (config.KERAS_DIRECTORY + "bestweightsfor_" + config.SAVE_MODEL_NAME)
    check_if_checkpoint_already_exists(checkpoint_path)
    model = build_and_compile_model(checkpoint_path)

    with tf.device('/GPU:0'):

        train_generator, test_generator = get_datagenerators()

        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')  
        
        history = model.fit(train_generator, epochs=config.EPOCHS, validation_data=test_generator, callbacks=[checkpoint_callback])

        model.load_weights(checkpoint_path)
        model.save(config.KERAS_DIRECTORY + config.SAVE_MODEL_NAME)

        _, test_acc = model.evaluate(test_generator, verbose=0) # returns test_loss, test_acc but test_loss is not needed

        utils.print_and_plot_results(history, test_acc)



if __name__ == '__main__':
    train()

