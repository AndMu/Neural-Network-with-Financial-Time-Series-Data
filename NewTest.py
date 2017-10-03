import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
import seaborn as sns; sns.set()
from keras import callbacks
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
from keras.utils import np_utils
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.optimizers import RMSprop
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
neurons = [256, 256, 32, 3]
dropout = 0.3
decay = 0.5
epochs = 90
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

    return df


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    print ("Amount of features = {}".format(amount_of_features))
    data = stock.as_matrix()
    sequence_length = seq_len + 1 # index starting from 0
    result = []

    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days

    result = np.array(result)
    row = round(0.8 * result.shape[0]) # 80% split
    print ("Amount of training data = {}".format(0.9 * result.shape[0]))
    print ("Amount of testing data = {}".format(0.1 * result.shape[0]))

    train = result[:int(row), :] # 90% date
    X_train = train[:, :-1] # all data until day m
    y_train = train[:, -1][:,-1] # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]


def build_model(shape, neurons, dropout, decay):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(shape[0], shape[1]), return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(neurons[1], input_shape=(shape[0], shape[1]), return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
    if neurons[3] == 1:
        model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
        adam = keras.optimizers.Adam(decay=decay)
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    else:
        model.add(Dense(neurons[3]))
        model.add(Activation('softmax'))
        model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

    model.summary()
    return model

def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

df = get_stock_data(stock_name, ma=[50, 100, 200])
train_df = df.loc[df.index < '2017-01-01']
train_df = set_signals(train_df)
# train_df.to_csv("data2.csv")
test_df = df.loc[df.index > '2017-01-01']
test_df = set_signals(test_df)

X_train, y_train, X_test, y_test = load_data(train_df, seq_len)

y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)

model = build_model(shape, neurons, dropout, decay)

cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]
model.fit(
    X_train,
    y_train,
    batch_size=800,
    callbacks=cbks,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
    shuffle=True)

model.save_weights("trained.h5")
model_score(model, X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = load_data(test_df, seq_len)
result_y_prob1 = model.predict_proba(X_train, batch_size=800, verbose=1)
result_y_prob2 = model.predict_proba(X_test, batch_size=800, verbose=1)