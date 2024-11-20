import os
import cv2
import PIL
import pandas as pd
import numpy as np

cv2.setNumThreads(1)
here = os.path.dirname(__file__)

focusing_finding = ['No Finding', 
                    'Pneumothorax', 
#                     'Pneumomediastinum', 'Pneumopericardium',
                    'Mass', 'Nodule', 'Mediastinal Mass', 
                    'Lung Opacity', 
                    'Pleural Effusion', 
                    'Atelectasis', 
#                     'Airway Narrowing', 
                    'Tracheal-Mediastinal Shift', 'Volume Loss', 
                    'Osteolytic Lesion', 'Fracture', 'Sclerotic Lesion', 
#                     'Air-filled Space', 
                    'Cardiomegaly', 
                    'Bronchiectasis',
#                     'Increased Lung Volume'
                   ]


# chestxray's
MEAN = [0.4984]
SD = [0.2483]

def eval_transform(image, size = 1024):

    if len(image.shape) > 1 :
        image = PIL.Image.fromarray(np.uint8(image)).convert('L') # Convert the image to grayscale
        image = np.array(image)
    
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = Normalize(image, MEAN, SD)
    return image

def Normalize(image, MEAN, SD):
    # https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
    max_pixel_value = image.max()
    output = (image - MEAN[0] * max_pixel_value) / (SD[0] * max_pixel_value)
    return output

def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img
