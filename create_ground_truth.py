import cv2
import numpy as np
import operator
from pprint import pprint as pp

img = cv2.imread("./groundTruth.tif")
originalImage = cv2.imread("./originalImage.tif")
gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
print(img.dtype)

rows, cols, channels = img.shape

n = 7
offset = int(n/2)

for row in range(offset, rows - offset): 
    for col in range(offset, cols - offset):
        gtArea = img[row-offset:row+offset+1, col-offset:col+offset+1, 0]
        area = originalImage[row-offset:row+offset+1, col-offset:col+offset+1, :]
        gray_area = gray_image[row-offset:row+offset+1, col-offset:col+offset+1]

        unique, counts = np.unique(gray_area, return_counts=True)
        dict_gray_counts = dict(zip(unique, counts))
        gray_label = max(dict_gray_counts.items(), key=operator.itemgetter(1))[0]
  


        unique, counts = np.unique(gtArea, return_counts=True)
        dict_counts = dict(zip(unique, counts))
        label = max(dict_counts.items(), key=operator.itemgetter(1))[0]
        if gray_label == 0 or gray_label == 255:
            print("skipped")
        else:
            if label == 255:
                print("matrix")
                cv2.imwrite("./train{}x{}/matrix/{}_{}.png".format(n, n, int(row), int(col)), area)
            else:
                print("clast")
                cv2.imwrite("./train{}x{}/clast/{}_{}.png".format(n, n, int(row), int(col)), area)
