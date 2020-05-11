import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

#Global variables
BATCH_SIZE = 256            #Slice the data into batches
BUFFER_SIZE = 10000         # Buffer size to shuffle the dataset
                            # (TF data is designed to work with possibly infinite sequences,
                            # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
                            # it maintains a buffer in which it shuffles elements).
EVALUATION_INTERVAL = 200
EPOCHS = 10                 #Iterate over the dataset

past_history = 720  #Days to analyze
future_target = 72  #Days to predict
STEP = 6

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

def create_time_steps(length):
  return list(range(-length, 0))

def baseline(history):
  return np.mean(history)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
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
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  plt.show()
  return plt

def readFile(fileName):
    # Read file
    df = pd.read_csv(fileName, sep=",", header=0)
    return df

def singleStepModel(x_train_single, y_train_single, x_val_single, y_val_single):
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32,
                                               input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)
    plot_train_history(single_step_history,
                       'Single Step Training and validation loss')


def main():
    fileName = "price-volume-data/Data/Stocks/a.us.txt"
    df=readFile(fileName)
    features_considered = ['Open','High','Low','Close','Volume','OpenInt']
    features = df[features_considered]
    features.index = df['Date']

    features.plot(subplots=True)
    plt.show()

    #Where to split
    TRAIN_SPLIT = int(len(df)*0.9)
    print(TRAIN_SPLIT)

    #Split and normalize
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    #dataset = (dataset - data_mean) / data_std
    dataset = tf.keras.utils.normalize(
        dataset, axis=0
    )

    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)



    #Single step model
    singleStepModel(x_train_single, y_train_single, x_val_single, y_val_single)







main()