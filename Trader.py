from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import pandas as pd


class TestStrategy(bt.Strategy):

    params = (
        ('signal_file', ''),
    )

    def __init__(self):
        self.df = pd.read_csv(self.params.signal_file)
        self.df['date'] = pd.to_datetime(self.df['date'])

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])
        # self.sell()
        date = self.datas[0].datetime.date(0)
        filtered = self.df[self.df["date"] == date]
        if len(filtered) == 0:
            return
        actual_signal = filtered.iloc[0]["Signal"]
        if pd.isnull(actual_signal):
            return
        if actual_signal.lower() == "buy":
            self.buy()
        elif actual_signal.lower() == "sell":
            self.sell()


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy, signal_file='data2.csv')

    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    data = bt.feeds.Quandl(dataname='AMD', apikey='XH28RzhxDVHKWwnaN1Hv')
    cerebro.adddata(data)
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()

