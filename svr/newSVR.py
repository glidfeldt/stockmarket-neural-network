
#imports
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR # Support Vector Regression model
#rom sklearn.model_selection import train_test_split
import os


def newSVR(df):
    #Load the data - stock data from YAHOO
    #from google.colab import files
    #uploaded = files.upload()

    #fileName = "../price-volume-data/Data/Stocks/a.us.txt"
    #df = pd.read_csv(fileName)
    df.head(10)

    #Create the lists / X (independent) and Y (dependent) dataset
    dates = []
    prices = []

    # Get rows and cols from dataset
    df.shape

    # print last row of data (this will be the test data)

    train_split= int(len(df) / 2)
    target_row = df.iloc[[train_split]]
    target_data = target_row.loc[:,'Close']
    #print(target_data)
    for elem in target_data:
        target = elem

    #Split data
    df = df.head(len(df)-train_split)
    #print(df.shape)

    df_dates = df.loc[:,'Date'] # all rows from Date col
    df_open = df.loc[:,'Close'] # all rows from Open col

    #Create the independent data set 'X' as dates
    count = 1
    for date in df_dates:
      #dates.append([int(date.split('-')[2])])
      #dates.append([int(date.replace('-', ''))])
        dates.append([count])
        count = count+1
    #print(count)

    #Create dependent dataset 'Y' as prices
    for close_price in df_open:
      prices.append(float(close_price))
    #print(dates)
    #print(prices)

    #See what days where recorded in the dataset
    #print(dates)

    #Predict price of stock on given (last or similar) day
    predicted_price = predict_prices(dates,prices,[[82]])
    print('Prediction: '+str(predicted_price))
    print('Target: '+str(target))


# Three different SVR models with different kernels
def predict_prices(dates, prices, x):
    print("started..")

    #Create 3 SVR models C = error term
    #Train models in the dates and prices
    '''
    # Linear kernel
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_lin.fit (dates, prices)
    print("finished linear..")

    # Polynomial Kernel
    svr_poly = SVR(kernel='poly', C=1e3, degree=4, gamma='scale')
    svr_poly.fit(dates, prices)
    print("finished poly..")
    '''
    # Radial Basis Kernel
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale')
    svr_rbf.fit (dates, prices)
    print("finished rbf..")

    # Printable results
    #result_lin = str(round(svr_lin.predict(x)[0],2))
    #result_poly= str(round(svr_poly.predict(x)[0],2))
    result_rbf = str(round(svr_rbf.predict(x)[0],2))

    # Plot preset
    plt.figure(dpi=600)

    #Plot models on a graph the see which one has the best fit
    plt.scatter(dates,prices,color = 'black', label='Target: '+str(x))
    #plt.scatter(dates,svr_lin.predict(dates),color = 'green', label='Linear: '+result_lin)
    #plt.scatter(dates,svr_poly.predict(dates),color = 'red', label='Poly: '+result_poly)
    plt.scatter(dates,svr_rbf.predict(dates),color = 'blue', label='RBF: '+result_rbf)

    plt.title('Support Vector Regression models')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    #plt.savefig('SVR-plot.png', dpi=600)

    #return all three model predictions
    return svr_rbf.predict(x)[0]\
        #, svr_lin.predict(x)[0], svr_poly.predict(x)[0]


#newSVR()