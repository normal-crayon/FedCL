#!usr/bin/myenv python
# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PsmhBpW2cmXfonSDDJSvyuo_UUCWwJpk
"""

from operator import mod
import tensorflow as tf
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import pickle
import time

from tensorflow import keras
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.ops.gen_array_ops import shape
import flwr as fl
# import tensorflow_datasets as tfds
with open('my_dataset.pickle', 'rb') as input:
  (x_train, x_test), (y_train, y_test) = pickle.load(input)

labels = np.unique(y_test)
hist = []




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

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

class SvhnClient(fl.client.NumPyClient):
    def get_parameters(self):
          return model.get_weights()
    
    def fit(self, parameters, config):
          model.set_weights(parameters)
          r = model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=1)
          hist.append(r.history)
          with open('client2.pickle', 'wb') as metrics:
                pickle.dump(hist, metrics)
          return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
          model.set_weights(parameters)
          loss, accuracy = model.evaluate(x_test, y_test)
          return loss, len(x_test), {'accuracy': accuracy}
# r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

fl.client.start_numpy_client("localhost:5000", client=SvhnClient())