import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
# The code below makes sure that your results are reproducible (tightening the randomness) 
# (you can change the seed number to your liking), Note that optimzers are stochastic nature
seed(17)
import keras

from keras.datasets import mnist
(train_image,train_label),(test_image,test_label) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
# We transform 28x28 images to a vector of 1x768
X_train = train_image.reshape(60000,784)
X_test = test_image.reshape(10000,784)

# We transform labels images to a vector of label 
# (e.g., 5 becomes [0,0,0,0,1,0,0,0,0])
# (e.g., 0 becomes [1,0,0,0,0,0,0,0,0])
y_train = to_categorical(train_label)
y_test = to_categorical(test_label)

from keras import layers, models
# The model is sequential, meaning that we can add a sequence of layers
# Adding a sequence can be done by network.add() 
network = models.Sequential()
network.add(layers.Dense(128, activation = 'sigmoid')) #Layer 1
network.add(layers.Dense(10, activation = 'sigmoid')) #Last layer must have 10 neuron (each neuron represent a digit)

# Configures the model for training.
network.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
# Trains the model for a fixed number of epochs
history = network.fit(X_train,y_train, epochs=20, batch_size=16)

# Returns the loss value & metrics values for the model in test mode.
network.evaluate(X_test, y_test, batch_size=1)

plt.rcParams['figure.figsize'] =(4,4)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy - percentage (0-1)')
plt.xlabel('epoch')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
