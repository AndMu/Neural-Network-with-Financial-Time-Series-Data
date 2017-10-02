import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
import seaborn as sns; sns.set()
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import pandas_datareader.data as web
import h5py
import quandl

quandl.ApiConfig.api_key = 'XH28RzhxDVHKWwnaN1Hv'
seq_len = 22
shape = [seq_len, 9, 1]
neurons = [256, 256, 32, 1]
dropout = 0.3
decay = 0.5
epochs = 90
stock_name = 'AMD'


def get_stock_data(stock_name, normalize=True, ma=[]):
    """
    Return a dataframe of that stock and normalize all the values.
    (Optional: create moving average)
    """
    df = quandl.get_table('WIKI/PRICES', ticker=stock_name)

    df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend', 'volume', 'split_ratio'], 1, inplace=True)
    df.set_index('date', inplace=True)

    # Renaming all the columns so that we can use the old version code
    df.rename(columns={'adj_open': 'Open', 'adj_high': 'High', 'adj_low': 'Low', 'adj_volume': 'Volume',
                       'adj_close': 'Adj Close'}, inplace=True)

    # Percentage change
    df['Pct'] = df['Adj Close'].pct_change()
    flipped_low = df['Low'].iloc[::-1]
    flipped_high = df['High'].iloc[::-1]
    df["Low_Mov"] = flipped_low.rolling(5, 1).min()
    df["High_Mov"] = flipped_high.rolling(5, 1).max()
    df.to_csv("data.csv")
    df.dropna(inplace=True)

    # Moving Average
    if ma != []:
        for moving in ma:
            df['{}ma'.format(moving)] = df['Adj Close'].rolling(window=moving).mean()
    df.dropna(inplace=True)

    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Volume'] = min_max_scaler.fit_transform(df.Volume.values.reshape(-1, 1))
        df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
        df['Pct'] = min_max_scaler.fit_transform(df['Pct'].values.reshape(-1, 1))
        if ma != []:
            for moving in ma:
                df['{}ma'.format(moving)] = min_max_scaler.fit_transform(
                    df['{}ma'.format(moving)].values.reshape(-1, 1))

                # Move Adj Close to the rightmost for the ease of training

    return df


df = get_stock_data(stock_name, ma=[50, 100, 200])