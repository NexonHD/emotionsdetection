import config
import numpy as np
import matplotlib.pyplot as plt

def get_data(set: str = 'train'):
    DATA_LOCATION = config.NPZ_DATA_LOCATION

    mergeddata_dict = np.load(DATA_LOCATION)

    train_images = mergeddata_dict['0']
    train_labels = mergeddata_dict['1']
    test_images = mergeddata_dict['2']
    test_labels = mergeddata_dict['3']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = np.transpose(train_images, (2,0,1))
    test_images = np.transpose(test_images, (2,0,1))
    
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    return (train_images, train_labels, test_images, test_labels)

def print_and_plot_results(history, test_acc):
    print(f'\n\n------------------------------------------------------------------- \n\n *** Finale Trefferquote auf Testdaten: {test_acc}\n\n-------------------------------------------------------------------\n\n')

    plt.plot(history.history['accuracy'], label='Trainingsdaten')
    plt.plot(history.history['val_accuracy'], label='Testdaten')
    plt.xlabel('Epoche'), plt.ylabel('Trefferquote'), plt.legend(loc='lower right')
    plt.show()