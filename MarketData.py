from datetime import timedelta

import quandl
from sklearn import preprocessing
import pandas as pd
quandl.ApiConfig.api_key = 'XH28RzhxDVHKWwnaN1Hv'


class MarketData(object):

    def __init__(self, stock_name, df, ma=[], indicator=None):
        self.scalers = {}
        self.stock_name = stock_name,
        self.df = df
        self.ma = ma
        self.indicator = indicator

    def get_stock_data(self, normalize=True):
        df = self.df.copy()
        # Percentage change
        df['Pct'] = df['Adj Close'].pct_change()

        # Moving Average
        if self.ma != []:
            for moving in self.ma:
                df['{}ma'.format(moving)] = df['Adj Close'].rolling(window=moving).mean()

        indicator_columns = []
        if self.indicator is not None:
            df_signal = self.indicator.calculate(df)
            df = df.join(df_signal)
            indicator_columns = list(df_signal.columns.values)

        df.dropna(inplace=True)
        if normalize:
            self.normalize_value(df, 'Open')
            self.normalize_value(df, 'High')
            self.normalize_value(df, 'Low')
            self.normalize_value(df, 'Volume')
            self.normalize_value(df, 'Adj Close')
            self.normalize_value(df, 'Pct')
            if self.ma != []:
                for moving in self.ma:
                    name = '{}ma'.format(moving)
                    self.normalize_value(df, name)

            for signal in indicator_columns:
                self.normalize_value(df, signal)

                    # Move Adj Close to the rightmost for the ease of training
        adj_close = df['Adj Close']
        df.drop(labels=['Adj Close'], axis=1, inplace=True)
        df = pd.concat([df, adj_close], axis=1)

        return df

    def insert_value(self, price):
        last_date = self.df.index[-1] + timedelta(days=1)
        self.df.ix[last_date] = self.df.ix[self.df.index[-1]]
        self.df.ix[last_date]['Adj Close'] = price
        self.df.ix[last_date]['Open'] = price
        self.df.ix[last_date]['High'] = price * 1.005
        self.df.ix[last_date]['Low'] = price * 0.995

    def normalize_value(self, df, name):
        values = df[name].values.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.scalers[name] = min_max_scaler
        df[name] = min_max_scaler.fit_transform(values)

    def denormalize(self, name, normalized_value):
        array_data = normalized_value.reshape(-1, 1)
        min_max_scaler = self.scalers[name]
        new = min_max_scaler.inverse_transform(array_data)
        return new


class MarketDataSource(object):

    def get_stock_data_basic(self, stock_name):
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

        df.dropna(inplace=True)
        return df