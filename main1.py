from newRNN import *
from svr.newSVR import *
import pandas as pd
import csv
import os

if __name__ == "__main__":
    x_axis = list(range(-20, 10))
    print(x_axis[-1:])
    filenames = ["a.us.txt", "aa.us.txt", "aaap.us.txt"]
    rmse_rnn=[]
    rmse_svr=[]
    tot_svr=0
    tot_rnn=0
    for file in filenames:
        dirname, _ = os.path.split(os.path.abspath(__file__))
        fileName = dirname+"/price-volume-data/Data/Stocks/"+file
        print(fileName)
        df = pd.read_csv(fileName)
        #SVR
        rmse=newSVR(df)
        rmse_svr.append(rmse)
        tot_svr+=rmse


        #RNN
        rmse =rnnNew()
        rmse_svr.append(rmse)
        tot_rnn +=rmse

    print("-----SVR-------")
    print(rmse_svr)
    tot_svr
    print("Total: +")
