import numpy as np
import pandas as pd
import seaborn as sns;

import Constants
from MarketData import MarketData
from DataProcessing import DataProcessing
from PredictionModel import PredictionModel

sns.set()
from keras.utils import np_utils

import quandl

quandl.ApiConfig.api_key = 'XH28RzhxDVHKWwnaN1Hv'
seq_len = 22
shape = [seq_len, 9, 1]
neurons = [256, 256, 32, 3]
dropout = 0.3
decay = 0.5

stock_name = 'AMD'


def set_signals(df):
    # Percentage change
    flipped_low = df['Low'].iloc[::-1]
    flipped_high = df['High'].iloc[::-1]
    df["Low_Mov"] = flipped_low.rolling(5, 1).min()
    df["High_Mov"] = flipped_high.rolling(5, 1).max()
    df["Signal_Text"] = ""
    df["Signal"] = 0
    df.loc[df["Low_Mov"] == df["Low"], "Signal_Text"] = "BUY"
    df.loc[df["Low_Mov"] == df["Low"], "Signal"] = 1
    df.loc[df["High_Mov"] == df["High"], "Signal_Text"] = "SELL"
    df.loc[df["High_Mov"] == df["High"], "Signal"] = 2

    df.drop(labels=['Signal_Text'], axis=1, inplace=True)
    df.drop(labels=['Low_Mov'], axis=1, inplace=True)
    df.drop(labels=['High_Mov'], axis=1, inplace=True)

    signal = df['Signal']
    df.drop(labels=['Signal'], axis=1, inplace=True)
    df.drop(labels=['Adj Close'], axis=1, inplace=True)
    df = pd.concat([df, signal], axis=1)
    return df


df = MarketData.get_stock_data(stock_name, ma=[50, 100, 200])
train_df = df.loc[df.index < '2017-01-01', :].copy()
train_df = set_signals(train_df)
# train_df.to_csv("data2.csv")

X_train, y_train, X_test, y_test = DataProcessing.load_data(train_df, seq_len)

y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)

model = PredictionModel(shape, neurons, dropout, decay)

model.fit(X_train, y_train)

model.model.save_weights("trained.h5")
model.model_score_class(X_test, y_test)

test_df = df.loc[df.index >= '2017-01-01', :].copy()
test_df = set_signals(test_df)

X_train, y_train, X_test, y_test = DataProcessing.load_data(test_df, seq_len)
result_y_prob1 = model.model.predict_proba(X_train, batch_size=Constants.TEST_BATCH, verbose=1)
result_y_prob2 = model.model.predict_proba(X_test, batch_size=Constants.TEST_BATCH, verbose=1)

zeros = np.zeros((seq_len + 1, result_y_prob2.shape[1]))
x = np.vstack((result_y_prob1, result_y_prob2, zeros))
for i in range(0, result_y_prob2.shape[1]):
    test_df["Result-" + str(i)] = x[:, i]

test_df.to_csv("result.csv")

