#pip install pillow
#pip install numpy
from PIL import Image
import numpy
import os as os
import time

import config

DATA_NAME = config.DATA_NAME
DATASET_PATH = config.DATASET_PATH
LABELS = config.LABELS
SUBDATASETS = config.SUBDATASETS
KEEP_TEMP_FILES = config.KEEP_TEMP_FILES

def get_image_data_array(datasetpath, subdataset):
    #dataset compiler (dont run, takes a lot of time estimate: 30 minutes) import from .npy instead this is only for creation of .npy files
    data = numpy.zeros((1,48,48))
    i = 0

    #loop through directories in path
    for directory in os.listdir(datasetpath + subdataset):
        
        #loop through files in directory
        for file in os.listdir((datasetpath + subdataset + '/' + directory)):
            #create 2d array from img
            image = Image.open((datasetpath + subdataset + '/' + directory + "/" + file))
            imgarr = numpy.asarray(image)
            
            #append image (2d array) to data (3d array in format [picture index][x][y])
            data = numpy.vstack((data, imgarr[None]))
            i = i+1
            if (i%100==0):
                print(i)

    train_images = numpy.transpose(train_images, (1,2,0))

    #delete wrong item at index=0
    data = numpy.delete(data, 0, axis=2)
    return data

def get_labels_array(datasetpath, subdataset):
    labels = numpy.empty([0])

    currentlabel = 0
    for directory in os.listdir(f'{datasetpath}{subdataset}'):
        for _ in range(len(os.listdir(f'{datasetpath}{subdataset}/{directory}'))):
            labels = numpy.append(labels, currentlabel)
        currentlabel = currentlabel + 1

    print(f'{subdataset} labels shape {labels.shape}')
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
    numpy.save(f'{path}temp/shuffled_{subdataset}data.npy', shuffled_data)
    numpy.save(f'{path}temp/shuffled_{subdataset}labels.npy', shuffled_labels)

def merge(subdatasets, path):
    mergeddata_dict = {}
    dict_index = 0

    for subdataset in subdatasets:
        data = numpy.load(f'{path}temp/shuffled_{subdataset}data.npy')
        mergeddata_dict[str(dict_index)] = data
        dict_index = dict_index + 1
    
        labels = numpy.load(f'{path}temp/shuffled_{subdataset}labels.npy')
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
    merged = merge(subdatasets, path)
    save((f'{path}/{dataname}'), merged)

    if keepTempFiles == False:
        for subdataset in subdatasets:
            os.remove(f'{path}shuffled_{subdataset}data.npy')
            os.remove(f'{path}shuffled_{subdataset}labels.npy')

    duration = time.time() - duration
    print(f'({round(duration, 2)}s) dataset saved as {dataname}.npz to {path}')

compileDataset(SUBDATASETS, DATASET_PATH, DATA_NAME, KEEP_TEMP_FILES)
