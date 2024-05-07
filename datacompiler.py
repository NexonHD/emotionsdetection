#pip install opencv-python
#pip install pillow
#pip install numpy
import cv2
from PIL import Image
import numpy
import os as os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
#from tensorflow.keras import datasets, layers, models

DATA_NAME = 'data'
DATASET_PATH = 'C:/Users/linie/vsc/emotionsdetection/data/'
LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
SUBDATASETS = ["train", "test"]
KEEP_TEMP_FILES = False

def get_image_data_array(datasetpath, subdataset):
    i = 0

    image_list = []

    for directory in os.listdir(datasetpath + subdataset):
        for file in os.listdir((datasetpath + subdataset + '/' + directory)):
            image = Image.open((datasetpath + subdataset + '/' + directory + "/" + file))
            imgarr = numpy.asarray(image)
            image_list.append(imgarr)
            i = i+1
            if(i%(100)==0): print(i)

    data = numpy.array(image_list)

    return data

def get_labels_array(datasetpath, subdataset):
    labels = numpy.empty([0])

    currentlabel = 0
    for directory in os.listdir(datasetpath + subdataset):
        for fileindex in range(len(os.listdir(datasetpath + subdataset + '/' +directory))):
            labels = numpy.append(labels, currentlabel)
        currentlabel = currentlabel + 1
    return labels

def get_random_order(data):
    randomorder = numpy.empty(shape=(0,))
    for n in range(numpy.shape(data)[2]):
        randomorder = numpy.append(randomorder, n)

    numpy.random.shuffle(randomorder)
    return randomorder

def shuffle(data, labels, randomorder):
    shuffled_data = numpy.copy(data)
    for n in range(len(randomorder)):
        randomindex = randomorder[n]
        shuffled_data[:,:,int(randomindex)] = data[:,:,int(n)]
    
    shuffled_labels = numpy.copy(labels)
    for realindex in range(len(randomorder)):
        randomindex = randomorder[realindex]
        shuffled_labels[int(randomindex)] = labels[int(realindex)]
    return shuffled_data,shuffled_labels

def save_npy(subdataset, shuffled_data, shuffled_labels, path):
    numpy.save(f'{path}shuffled_{subdataset}data.npy', shuffled_data)
    numpy.save(f'{path}shuffled_{subdataset}labels.npy', shuffled_labels)

def load_npy(subdataset):
    shuffled_data = numpy.load('shuffled_' + subdataset + 'data.npy')
    shuffled_labels = numpy.load('shuffled_' + subdataset + 'labels.npy')

def merge(subdatasets):
    mergeddata_dict = {}
    dict_index = 0

    for subdataset in subdatasets:
        data = numpy.load('shuffled_' + subdataset + 'data.npy')
        mergeddata_dict[str(dict_index)] = data
        dict_index = dict_index + 1
    
        labels = numpy.load('shuffled_' + subdataset + 'labels.npy')
        mergeddata_dict[str(dict_index)] = labels
        dict_index = dict_index + 1
    return mergeddata_dict

def save(dataname, mergeddata_dict):
    numpy.savez(dataname, **mergeddata_dict)

def load(dataname):
    mergeddata_dict = numpy.load(f'{dataname}.npz')
    return mergeddata_dict


def compileDataset(subdatasets, path, dataname, keepTempFiles):
    duration = time.time()
    for subdataset in subdatasets:
        sorted_data = get_image_data_array(path, subdataset)
        sorted_labels = get_labels_array(path, subdataset)
        data, labels = shuffle(sorted_data, sorted_labels, get_random_order(sorted_data))
        save_npy(subdataset, data, labels, DATASET_PATH)
    merged = merge(subdatasets)
    save((path + '/' + dataname), merged)

    for subdataset in subdatasets:
        os.remove(f'{path}shuffled_{subdataset}data.npy')

    duration = time.time() - duration
    print(f'{duration} saved as {dataname}.npz to {path}')

compileDataset(SUBDATASETS, DATASET_PATH, DATA_NAME, KEEP_TEMP_FILES)
