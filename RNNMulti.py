import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt



class RNNMulti():
    def __init__(self, dataset,
                 train_split, past_history,
                 future_target, step,
                 epochs, evaluation_interval,
                 batch_size, buffer_size):

        self.dataset=dataset
        self.train_split=train_split
        self.past_history=past_history
        self.future_target=future_target
        self.step=step
        self.epochs=epochs
        self.evaluation_interval=evaluation_interval
        self.batch_size=batch_size
        self.buffer_size=buffer_size

        self.multi_step_model=None
        self.multi_step_history=None
        self.val_data_multi=None


        '''
        self.x_train_multi, self.y_train_multi = self.multivariate_data(self.dataset, self.dataset[:, 1], 0,
                                                         self.train_split, self.past_history,
                                                         self.future_target, self.step)
        self.x_val_multi, self.y_val_multi = self.multivariate_data(self.dataset, self.dataset[:, 1],
                                                     self.train_split, None, self.past_history,
                                                     self.future_target, self.step)
        '''
        self.x_train_uni, self.y_train_uni = self.univariate_data(self.dataset, 0, self.train_split,
                                                   self.past_history,
                                                   self.future_target)
        self.x_val_uni, self.y_val_uni = self.univariate_data(self.dataset, self.train_split, None,
                                                  self.past_history,
                                                  self.future_target)

        self.train_univariate = tf.data.Dataset.from_tensor_slices((self.x_train_uni, self.y_train_uni))
        self.train_univariate = self.train_univariate.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

        self.val_univariate = tf.data.Dataset.from_tensor_slices((self.x_val_uni, self.y_val_uni))
        self.val_univariate = self.val_univariate.batch(self.batch_size).repeat()

        self.simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(8, input_shape=self.x_train_uni.shape[-2:]),
            tf.keras.layers.Dense(1)
        ])

        self.simple_lstm_model.compile(optimizer='adam', loss='mae')

    def uni_train(self):
        self.simple_lstm_model.fit(self.train_univariate, epochs=self.epochs,
                              steps_per_epoch=self.evaluation_interval,
                              validation_data=self.val_univariate, validation_steps=50)

    def multivariate_data(self, dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i + target_size])
            else:
                labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)

    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + target_size])
        return np.array(data), np.array(labels)


    def univariate_data(self, dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + target_size])
        return np.array(data), np.array(labels)

    def train(self):
        train_data_multi = tf.data.Dataset.from_tensor_slices((self.x_train_multi, self.y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

        self.val_data_multi = tf.data.Dataset.from_tensor_slices((self.x_val_multi, self.y_val_multi))
        self.val_data_multi = self.val_data_multi.batch(self.batch_size).repeat()

        #for x, y in train_data_multi.take(1):
        #    self.multi_step_plot(x[0], y[0], np.array([0]))

        self.multi_step_model = tf.keras.models.Sequential()
        self.multi_step_model.add(tf.keras.layers.LSTM(32,
                                                  return_sequences=True,
                                                  input_shape=self.x_train_multi.shape[-2:]))
        self.multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
        self.multi_step_model.add(tf.keras.layers.Dense(72))

        self.multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

        self.multi_step_history = self.multi_step_model.fit(train_data_multi, epochs=self.epochs,
                                                  steps_per_epoch=self.evaluation_interval,
                                                  validation_data=self.val_data_multi,
                                                  validation_steps=50)

    def plot_train_history(self, history, title):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()

        plt.show()

    def result(self):
        rmse = []
        for x, y in self.val_data_multi.take(3):
            self.multi_step_plot(x[0], y[0], self.multi_step_model.predict(x)[0])
            rmse.append(sqrt(mean_squared_error(y[0], self.multi_step_model.predict(x)[0])))
        return rmse

    def create_time_steps(self, length):
        return list(range(-length, 0))

    def plot_train_history(self, history, title):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()

        plt.show()

    def baseline(self, history):
        return np.mean(history)

    def multi_step_plot(self, history, true_future, prediction):
        plt.figure(figsize=(12, 6))
        num_in = self.create_time_steps(len(history))
        num_out = len(true_future)

        plt.plot(num_in, np.array(history[:, 1]), label='History')
        plt.plot(np.arange(num_out) / self.step, np.array(true_future), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out) / self.step, np.array(prediction), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.show()

    def show_plot(self, plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = self.create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10,
                         label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel('Time-Step')
        plt.show()
        #return plt












'''

def main():



    #print('Single window of past history : {}'.format(x_train_single[0].shape))

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], np.array([0]))

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                              return_sequences=True,
                                              input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(72))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              validation_steps=50)
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


    #Calcuualte rmse

    for x, y in val_data_multi.take(3):
        rms = sqrt(mean_squared_error(y[0], multi_step_model.predict(x)[0]))
        print("RMS: "+str(rms))
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])









main()
'''