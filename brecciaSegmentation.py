import os
import cv2
import numpy as np

groundTruthFilePath = "groundTruth.tif"
originalImageFilePath = "originalImage.tif"

groundTruthImage = cv2.imread(groundTruthFilePath)
originalImage = cv2.imread(originalImageFilePath)

groundTruthHeight, groundTruthWidth, groundTruthChannels = groundTruthImage.shape
originalImageHeight, originalImageWidth, originalImageChannels = originalImage.shape

clastPixelCoordinates = []
matrixPixelCoordinates = []

for row in range (0, groundTruthHeight):
	for col in range (0, groundTruthWidth):
		if(groundTruthImage[row][col][0]) == 0:

			clastPixelCoordinates.append([row, col])

		elif (groundTruthImage[row][col][0]) == 255:

			matrixPixelCoordinates.append([row, col])

print ("Total clast pixels: {}".format(len(clastPixelCoordinates)))
print ("Total matrix pixels: {}".format(len(matrixPixelCoordinates)))

clastPixelCoordinates = np.array(clastPixelCoordinates)
matrixPixelCoordinates = np.array(matrixPixelCoordinates)

np.random.shuffle(clastPixelCoordinates)
np.random.shuffle(matrixPixelCoordinates)

x_test = []
y_test = []

x_val = []
y_val = []

trainPixels = int((len(clastPixelCoordinates))*0.10)

for clastPixel in clastPixelCoordinates:
	xTestRow = clastPixel[0]
	xTestCol = clastPixel[1]

	if all(255 == pixel for pixel in originalImage[xTestRow][xTestCol]):

		print("Background pixel, skipping")

	else: 	

		

		if (len(x_test) < trainPixels):

			x_test.append(originalImage[xTestRow][xTestCol])
			y_test.append([1, 0])

		else:

			x_val.append(originalImage[xTestRow][xTestCol])
			y_val.append([1,0])

	if (len(x_val) > trainPixels):

		break

for matrixPixel in matrixPixelCoordinates:
	xTestRow = matrixPixel[0]
	xTestCol = matrixPixel[1]

	if all(255 == pixel for pixel in originalImage[xTestRow][xTestCol]):

		print("Background pixel, skipping")

	elif all(0 == pixel for pixel in originalImage[xTestRow][xTestCol]):

		print("Text pixel, skipping")

	else: 	

		
		
		if (len(x_test) < 2*trainPixels):
			x_test.append(originalImage[xTestRow][xTestCol])
			y_test.append([0, 1])

		else:
			x_val.append(originalImage[xTestRow][xTestCol])
			y_val.append([0,1])

	if (len(x_val) > 2*trainPixels):
		
		break

x_test = np.array(x_test)
y_test = np.array(y_test)

x_val = np.array(x_val)
y_val = np.array(y_val)

print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

