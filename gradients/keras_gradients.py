# Example taken from:
# https://stackoverflow.com/questions/39561560/getting-gradient-of-model-output-w-r-t-weights-using-keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np


model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

outputTensor = model.output
listOfVariableTensors = model.trainable_weights
gradients = K.gradients(outputTensor, listOfVariableTensors)

trainingExample = np.random.random((1, 8))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
evaluated_gradients = sess.run(gradients, feed_dict={model.input: trainingExample})
