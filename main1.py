from newRNN import *
from svr.newSVR import *
import pandas as pd
import csv
import os

if __name__ == "__main__":
    filenames = ["a.us.txt", "aa.us.txt", "aaap.us.txt"]
    rmse_rnn=[]
    rmse_rnnNew=[]
    for file in filenames:
        dirname, filename = os.path.split(os.path.abspath(__file__))
        fileName = dirname+"/price-volume-data/Data/Stocks/"+"file"
        df = pd.read_csv(fileName)
        newSVR(df)
        rnnNew()