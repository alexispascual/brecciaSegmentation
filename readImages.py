import os
import cv2
import numpy as np

from sklearn.utils import shuffle
from keras.utils import np_utils

#This function reads images from a directory
def readTrainingData(directory, imageLength, imageWidth, channels):

    print('Reading Dataset from {}'.format(directory))
    images = []
    labels = []
    currentClass = 0
    currentImageCount = 0

    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            image = cv2.imread(os.path.join(dirpath, file))
            #image = cv2.resize(image, (imageLength, imageWidth),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            labels.append(currentClass)
            currentImageCount = currentImageCount + 1

            if currentImageCount > 14000:
                break

        print("{} Images in Class {}".format(currentImageCount, currentClass))
        currentImageCount = 0
        currentClass += 1

    images, labels = shuffle(images, labels)
    images = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels - 1, currentClass - 1)

    print('Done reading dataset with {} classes and {} images'.format(currentClass - 1, len(images)))

    # print ("Saving Images and Labels to files")

    # imagesFileName = "LeiDatasetImages" + "(" + str (imageLength) + ")" + ".npy"
    # labelsFileName = "LeiDatasetLabels" + "(" + str (imageLength) + ")" + ".npy"

    # np.save(imagesFileName, images)
    # np.save(labelsFileName, labels)

    return images, labels
