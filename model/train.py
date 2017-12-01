import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# variables to be used to fit the model
batch_size = 32
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# MNIST images only have a depth of 1, must explicitly declare that.
# transform dataset from having shape (n, width, height) to (n, depth, width, height).
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	# that corresponds to  the (depth, width, height) of each digit image
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	# that corresponds to  the (depth, width, height) of each digit image
    input_shape = (img_rows, img_cols, 1)

#convert data type to float32 and normalize data values to the range [0, 1].
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# The y_train and y_test data are not split into 10 distinct class labels, 
# but rather are represented as a single array with the class values.
# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# declaring a sequential model format:
model = Sequential()
# declare the input layer
				# The first 3 parameters correspond to the number of convolution filters to use, the number of rows in each convolution kernel, and the number of columns in each convolution kernel
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
				 # input shape parameter should be the shape of 1 sample
                 input_shape=input_shape))
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

# compile the model and it will be ready to be trained
model.compile(loss=keras.losses.categorical_crossentropy,
				# Adam optimizer has a stepsize to be tuned but adadelta doesn't need any.
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# To fit the model, we have to declare the batch size and number of epochs to train for and pass the data in
model.fit(x_train, y_train,
		# use pre defined variables
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
		  
# evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save weights to be used later to predict
model.save_weights('weights.h5')