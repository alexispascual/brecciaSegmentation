import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras import optimizers

# Import Callbacks
from keras.callbacks import ModelCheckpoint

# Import Image generator
from keras.preprocessing.image import ImageDataGenerator

from readImages import readTrainingData

# Import Dataset Splitter
from sklearn.model_selection import train_test_split

datasetDirectory = "./train5x5"
n = 5
imageSize = 5
channels = 3
batch_size = 32
epochs = 10
images = None
labels = None
model = Sequential()

images, labels = readTrainingData(datasetDirectory, imageSize, imageSize, channels)

# Split the data into training and validation set
x_train, x_validation, y_train, y_validation = train_test_split(images, labels, test_size=0.3, random_state=np.random.randint(1000))

print (len(x_train))
print (len(y_train))
print (len(x_validation))
print (len(y_validation))

model.add(Conv2D(64, (n,n), activation='relu', input_shape=(n,n,channels)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Define Adam optimizer
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(
	optimizer=adam,
	loss='binary_crossentropy',
	metrics=['accuracy'])

trainingDatagen = ImageDataGenerator(
           	# 	rescale=1./255,
            #    rotation_range=20,
            #    width_shift_range=0.2,
            #    height_shift_range=0.2,
            #    horizontal_flip=True,
            #    vertical_flip=True,
            #    fill_mode="wrap",
            #    shear_range=0.2,
            #    zoom_range=0.2,
            #    validation_split = 0.3
            )

# Define model file name
modelFilePath = "convBrecciaSegmentation.h5"

checkpointer = ModelCheckpoint(filepath=modelFilePath, verbose=1, save_best_only=True)

trainingDatagen.fit(x_train)

# fits the model on batches with real-time data augmentation and save the model only if validation loss decreased
try:
    history = model.fit_generator(trainingDatagen.flow(
    	x_train, 
    	y_train, 
    	batch_size=batch_size),
        steps_per_epoch=len(x_train) / batch_size,
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        callbacks=[checkpointer])

except KeyboardInterrupt:
    print("Clearing session. Hopefully this works.")
    K.clear_session()
