from RNNMulti import *
from svr.svr import *
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


#Global variables
batch_size = 256            #Slice the data into batches
buffer_size = 10000         # Buffer size to shuffle the dataset
                            # (TF data is designed to work with possibly infinite sequences,
                            # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
                            # it maintains a buffer in which it shuffles elements).
evaluation_interval = 200
epochs = 10                 #Iterate over the dataset

past_history = 720          #Days to analyze
future_target = 72          #Days to predict
step = 6

def readFile(fileName):
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    # Read file
    df = pd.read_csv(fileName, sep=",", header=0)
    return df



if __name__ == "__main__":
    #Load data
    #d = dirname(dirname(abspath(__file__)))

    fileName="price-volume-data/Data/Stocks/a.us.txt"
    df=readFile(fileName)

    #Pick features to be used
    dataset = df['Close']
    dataset.index = df['Date']
    dataset = dataset.values

    # Split the data
    train_split = int(len(df) / 2)
    print(train_split)

    # Split and normalize
    data_mean = dataset[:train_split].mean()
    data_std = dataset[:train_split].std()
    dataset = (dataset - data_mean) / data_std


    #Initiate models
    rnnmulti = RNNMulti(dataset, train_split,
                        past_history, future_target,
                        step, epochs,
                        evaluation_interval, batch_size,
                        buffer_size)
    svr = SVR(dataset, train_split)

    #Train models
    rnnmulti.uni_train()
    svr_result = svr.train()

    #Predict

    #Result
    print(rnnmulti.result(svr_result))


    #Plot