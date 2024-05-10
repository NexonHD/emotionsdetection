import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
#%matplotlib inline

def get_cropped_image_if_2_eyes(img):
    #img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('emotionsdetection\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('emotionsdetection\haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(image_rgb, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_rgb[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color, (x, y, w, h)
    return None, None

        
def preprocess_image(img):
    img_only_face = get_cropped_image_if_2_eyes(img)
    if img_only_face is not None:
        try:
            numpydata = np.asarray(img_only_face)
            resizedImage = cv2.resize(numpydata, (48, 48))
            grayscaleImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
            return grayscaleImage
        except Exception as e:
            return None
    else:
        return None