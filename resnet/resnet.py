from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm
import time  #helper libraries
from os.path import dirname, abspath


#Step 1 Load Data
d = dirname(dirname(abspath(__file__)))
#print(d)
#X_train, y_train, X_test, y_test = lstm.load_data(d+'/price-volume-data/Data/Stocks/a.us.txt', 50, True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

