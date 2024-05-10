import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import config


DATA_LOCATION = config.NPZ_DATA_LOCATION
LOAD_MODEL_NAME = config.LOAD_MODEL_FROM_FILE
SAVE_MODEL_NAME = config.SAVE_MODEL_NAME

mergeddata_dict = numpy.load(DATA_LOCATION)

train_images = mergeddata_dict['0']
train_labels = mergeddata_dict['1']
test_images = mergeddata_dict['2']
test_labels = mergeddata_dict['3']

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = numpy.transpose(train_images, (2,0,1))
test_images = numpy.transpose(test_images, (2,0,1))

train_images = numpy.expand_dims(train_images, -1)
test_images = numpy.expand_dims(test_images, -1)

if LOAD_MODEL_NAME == '':
    model = config.create_model()
else:
    model = tf.keras.models.load_model(LOAD_MODEL_NAME)


with tf.device('/GPU:0'):

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

    #model = tf.keras.models.load_model("C:/Users/Yannick/Codes/Emotion Detector/models/2_0-SGD_optimizer-dataaugmentation.keras")

    adam_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    sgd_optimizer = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    checkpoint_path = (config.KERAS_DIRECTORY + "bestweightsfor_" + SAVE_MODEL_NAME)  
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')  

    model.compile(optimizer=adam_optimizer, metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(train_generator, epochs=200, validation_data=test_generator, callbacks=[early_stopping, checkpoint_callback])
    predictions = model.predict(test_images)

    model.load_weights(checkpoint_path)
    model.save(config.KERAS_DIRECTORY + SAVE_MODEL_NAME)

    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print('\n *** Finale Trefferquote auf Testdaten:', test_acc)

    plt.plot(history.history['accuracy'], label='Trainingsdaten')
    plt.plot(history.history['val_accuracy'], label='Testdaten')
    plt.xlabel('Epoche'), plt.ylabel('Trefferquote'), plt.legend(loc='lower right')
    plt.show()
