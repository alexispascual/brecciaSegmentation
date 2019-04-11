import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

# Import Callbacks
from keras.callbacks import ModelCheckpoint

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

	elif all(0 == pixel for pixel in originalImage[xTestRow][xTestCol]):

		print("Text pixel, skipping")

	else: 	

		if (len(x_test) < trainPixels):

			x_test.append(originalImage[xTestRow][xTestCol])
			y_test.append([1, 0])

		else:

			x_val.append(originalImage[xTestRow][xTestCol])
			y_val.append([1, 0])

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
			y_val.append([0, 1])

	if (len(x_val) > 2*trainPixels):
		
		break

x_test = np.array(x_test)
y_test = np.array(y_test)

#x_test = np.multiply(x_test, 1.0 / 255.0)

x_val = np.array(x_val)
y_val = np.array(y_val)

#x_val = np.multiply(x_val, 1.0 / 255.0)

print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

model = Sequential()

model.add(Dense(units=64, activation="tanh", input_dim=3))
#model.add(Dense(units=64, activation="tanh"))
model.add(Dense(units=2, activation="sigmoid"))

# Define Adam optimizer
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

checkpointer = ModelCheckpoint(filepath="modelBrecciaSegmentation.h5", verbose=1, save_best_only=True)

model.compile(
	optimizer=adam,
	loss='binary_crossentropy',
	metrics=['accuracy'])

model.fit(
	x_test,
	y_test, 
	epochs = 10, 
	batch_size=32,
	shuffle=True,
	validation_data=(x_val, y_val),
	callbacks=[checkpointer])




