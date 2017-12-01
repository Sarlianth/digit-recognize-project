import numpy as np
import keras.models
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def init():
    # variables to be used to for the model
    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28
    # that corresponds to  the (depth, width, height) of each digit image
    input_shape = (img_rows, img_cols, 1)
    # declaring a sequential model format:
    model = Sequential()
    # declare the input layer
    # The first 3 parameters correspond to the number of convolution filters to use, the number of rows in each convolution kernel, and the number of columns in each convolution kernel
    # input shape parameter should be the shape of 1 sample
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))	
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the previous layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout layer is a method for regularizing our model in order to prevent overfitting.
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # the final layer has an output size of 10, corresponding to the 10 classes of digits
    model.add(Dense(num_classes, activation='softmax'))
    
    # load weights into new model
    model.load_weights("model/weights.h5")

    # compile and evaluate loaded model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    # Returns the default graph being used in the current thread
    graph = tf.get_default_graph()

    # return model and graph
    return model, graph