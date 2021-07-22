#!usr/bin/myenv python

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.ops.gen_array_ops import shape
import flwr as fl
import pickle
import time

# fl.server.start_server(config={"num_rounds": 3})
hist = []
data = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0 , x_test/255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=x_train[0].shape),
            # tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu'),
            # tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu'),
            # tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu'),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
])
model.build(input_shape=x_train.shape)
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

class MnistClient(fl.client.NumPyClient):

  def get_parameters(self):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)
    hist.append(r.history)
    with open('client1_fit.pickle', 'wb') as metrics:
      pickle.dump(hist, metrics)
    return model.get_weights(), len(x_train), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, len(x_test), {'accuracy': accuracy }

fl.client.start_numpy_client("localhost:5000", client=MnistClient())