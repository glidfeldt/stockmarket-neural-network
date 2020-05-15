from newRNN import *
from svr.newSVR import *
import pandas as pd
import csv
import os

def cross_validation_rmse(list):
    return sum(list)/len(list)


if __name__ == "__main__":
    x_axis = list(range(-20, 10))
    print(x_axis[-1:])
    filenames = ["a.us.txt", "aa.us.txt", "aaap.us.txt"]
    rmse_rnn=[]
    rmse_svr=[]

    for file in filenames:
        dirname, _ = os.path.split(os.path.abspath(__file__))
        fileName = dirname+"/price-volume-data/Data/Stocks/"+file
        print(fileName)
        df = pd.read_csv(fileName)
        #SVR
        rmse=newSVR(df)
        rmse_svr.append(rmse)


        #RNN
        rmse =rnnNew()
        rmse_rnn.append(rmse)

    print("-----SVR-------")
    print(rmse_svr)
    print(cross_validation_rmse(rmse_svr))

    print("Total: +")
    print(rmse_rnn)
    print(cross_validation_rmse(rmse_rnn))
