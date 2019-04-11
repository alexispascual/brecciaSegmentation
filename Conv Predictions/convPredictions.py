# Predict whether the input image is a breccia or not
import numpy as np
import cv2
import os
from keras.models import load_model

n = 3

# Load Model
model = load_model('convBrecciaSegmentation3x3.h5')
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

trainDirectory = '../train' + str(n) + 'x' + str(n)
print("Training Directory: " + trainDirectory)

print("Generating Image")

for dirpath, dirnames, filenames in os.walk(trainDirectory):
	for filename in filenames:

		row, col = (filename.split(".")[0].split("_")[0], filename.split(".")[0].split("_")[1])
		imageToPredict = cv2.imread(os.path.join(dirpath, filename))
		imageToPredict = np.multiply(imageToPredict, 1.0/255.0)
		imageToPredict = imageToPredict.reshape(1, n, n, 3)
		
		prediction = model.predict(imageToPredict)

		if (np.argmax(prediction) == 1):
			blankImage[int(row), int(col)] = (255, 255 ,255)


print("Image Saved!")

cv2.imwrite('convPredicted.png',blankImage)