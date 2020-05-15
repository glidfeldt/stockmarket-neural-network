from newRNN import *
from svr.newSVR import *
import pandas as pd
import csv
import os

if __name__ == "__main__":
    dirname, filename = os.path.split(os.path.abspath(__file__))
    fileName = dirname+"/price-volume-data/Data/Stocks/a.us.txt"
    df = pd.read_csv(fileName)
    newSVR(df)
    rnnNew()