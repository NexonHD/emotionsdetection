import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay # type: ignore
import config
import utils
from sklearn.metrics import f1_score

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
        print('weights were not found')
    return model

def get_lr_schedule(type: str = 'exponential', initial_learning_rate = 0.01, end_lr = 0.000001, decay_change_steps = config.EPOCHS):
    decay_rate = (float(end_lr/initial_learning_rate))**(1.0/decay_change_steps)
    match type:
        case 'exponential':
            exponential_lr_schedule = ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=898*config.EPOCHS // decay_change_steps,  # Anpassung der decay_steps für  langsamere Reduktion
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


def get_custom_optimizer(optimizer: str = 'adam'):
    lr_schedule = get_lr_schedule()
    match optimizer:
        case 'adam':
            return keras.optimizers.Adam(learning_rate=lr_schedule)
        case 'sgd':
            return keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

def get_datagenerators():
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

    train_generator = datagen.flow(train_images, train_labels, batch_size=32)
    test_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

    return train_generator, test_generator

def train():
    checkpoint_path = (config.KERAS_DIRECTORY + "bestweightsfor_" + config.SAVE_MODEL_NAME)  
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
