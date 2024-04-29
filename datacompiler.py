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
from tensorflow.keras import datasets, layers, models

datasetpath = 'C:/Users/linie/Documents/Emotions Detection/data/'
dataname = 'test'
imgpath = r'C:\\Users\\linie\\Documents\\Emotions Detection\\scripts'
labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
datanames = ["train", "test"]

#image preprocessor
#leave path empty ('') if same path as the program
filename = 'sample.png'

inputImage = cv2.imread((imgpath+'\\'+filename))
imageArray = numpy.asarray(inputImage)
imageArray = cv2.resize(imageArray, (48, 48))
imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
outputImage = Image.fromarray(imageArray)
outputImage.save(imgpath+'\\'+'resized.png')

#dataset compiler (dont run, takes a lot of time estimate: 30 minutes) import from .npy instead this is only for creation of .npy files
data = numpy.zeros((1,48,48))
i = 0

#loop through directories in path
for directory in os.listdir(datasetpath + dataname):
    
    #loop through files in directory
    for file in os.listdir((datasetpath + dataname + '/' + directory)):
        #create 2d array from img
        image = Image.open((datasetpath + dataname + '/' + directory + "/" + file))
        imgarr = numpy.asarray(image)
        
        #append image (2d array) to data (3d array in format [picture index][x][y])
        data = numpy.vstack((data, imgarr[None]))
        i = i+1
        if (i%100==0):
            print(i)

#transpose 3d array to format [x][y][picture index]
data = numpy.transpose(data, (1,2,0))

#delete wrong item at index=0
data = numpy.delete(data, 0, axis=2)

#create an empty 1d array to store labels
labels = numpy.empty(shape=(0,))

currentlabel = 0
#loop through directories in path
for directory in os.listdir(datasetpath + dataname):
    #for every directory append n (size of directory) entries with value currentlabel to empty array
    for fileindex in range(len(os.listdir(datasetpath + dataname + '/' +directory))):
        labels = numpy.append(labels, currentlabel)
    currentlabel = currentlabel + 1

#create a random order to order both labels and data the same way but random
randomorder = numpy.empty(shape=(0,))

#get 1d array with numbers 0 through 28708 in order
for n in range(numpy.shape(data)[2]):
    randomorder = numpy.append(randomorder, n)

numpy.random.shuffle(randomorder)

#shuffle data according to randomorder
shuffled_data = numpy.copy(data)
for n in range(len(randomorder)):
    randomindex = randomorder[n]
    shuffled_data[:,:,int(randomindex)] = data[:,:,int(n)]
    
#shuffle labels according to randomorder
shuffled_labels = numpy.copy(labels)
for realindex in range(len(randomorder)):
    randomindex = randomorder[realindex]
    shuffled_labels[int(randomindex)] = labels[int(realindex)]

#save 3d data array as .npy in workingdirectory
#numpy.save(dataname + 'data.npy', data)
#numpy.save(dataname + 'labels.npy', labels)

numpy.save('shuffled_'+ dataname + 'data.npy', shuffled_data)
numpy.save('shuffled_'+ dataname + 'labels.npy', shuffled_labels)

#load data from <name>data.npy instead of creating a new 3d array
#data = numpy.load(dataname + 'data.npy')
#labels = numpy.load(dataname + 'labels.npy')

shuffled_data = numpy.load('shuffled_' + dataname + 'data.npy')
shuffled_labels = numpy.load('shuffled_' + dataname + 'labels.npy')

#merge .npy data into a .npz file

mergeddata_dict = {}
dict_index = 0

for dataname in datanames:
    data = numpy.load('shuffled_' + dataname + 'data.npy')
    mergeddata_dict[dict_index] = data
    dict_index = dict_index + 1
    
    labels = numpy.load('shuffled_' + dataname + 'labels.npy')
    mergeddata_dict[dict_index] = labels
    dict_index = dict_index + 1

# ** dictionary unpacking (here **arrays_dict = (arr0=0data, arr1=0labels, arr2=1data, arr3=1labels ...))
numpy.savez('data', **mergeddata_dict)

#dict = {
#    0 = shuffled_train_data
#    1 = shuffled_train_labels
#    2 = ...
#}

mergeddata_dict = numpy.load('data.npz')
