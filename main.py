#Question 1: Computer Vision 
#Classify the images in the given folder using the least number of images for training. Folder Names are the names of the classes
#Dataset URL: https://drive.google.com/drive/folders/1oQxSYEPV61fqmMcYbOba0i984q5a8Xsa?usp=sharing
#The code is written by Sneha Kumari

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__
import numpy as np

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:/HAILABS/train',
                                                 target_size = (64, 64),
                                                 batch_size = 2,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('D:/HAILABS/test',
                                            target_size = (64, 64),
                                            batch_size = 2,
                                            class_mode = 'categorical')

#y_test = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
y_test = np.array([0,0,1,1,2,2])
y_test = y_test.astype('int64')

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(training_set,
                  steps_per_epoch = 2,
                  epochs = 2,
                  validation_data = test_set,
                  validation_steps = 2)



predict_x=cnn.predict(test_set) 
classes_x=np.argmax(predict_x,axis=1)

classes_x=np.argmax(predict_x,axis=1)
from sklearn.metrics import precision_score

precision = precision_score(y_test,classes_x,average='micro')
print("Precision:" , precision)
