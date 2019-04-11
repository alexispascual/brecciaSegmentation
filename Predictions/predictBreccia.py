# Predict whether the input image is a breccia or not
import numpy as np
import cv2
import os
from keras.models import load_model

# Load Model
model = load_model('modelBrecciaSegmentation.h5')
print("Model loaded!")

# Constant definitions
imageFileName = 'originalImage.tif'

brecciaImage = cv2.imread(imageFileName)
print("Image successfully read!")

# Print Model Summary
print(model.summary())

# Read Each Pixel
imageHeight, imageWidth, imageChannels = brecciaImage.shape

# Blank Image
blankImage = np.zeros((imageHeight,imageWidth,3), np.uint8)

print("Generating Image")

for row in range (0, imageHeight):
	for col in range (0, imageWidth):
		
		inputPixels = brecciaImage[row][col]
		#inputPixels = np.multiply(inputPixels, 1.0 / 255.0)
		inputPixels = np.array(brecciaImage[row][col]).reshape((3,1))
		inputPixels = np.transpose(inputPixels)

		prediction = model.predict(inputPixels)

		if (np.argmax(prediction) == 1):
			blankImage[row,col] = (255, 255, 255)

		else:
			blankImage[row,col] = (0, 0, 0)

print("Image Saved!")

cv2.imwrite('predicted.png',blankImage)