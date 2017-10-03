from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import datetime, date, timedelta

import backtrader as bt
import pandas as pd
import backtrader.analyzers as btanalyzers

daysOffset = 10
last_date = date.today() - + timedelta(days=daysOffset)


class MySimpleStrategy(bt.Strategy):

    params = (
        ('signal_file', ''),
        ('security', ''),
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
        position = self.getpositionbyname(self.params.security).size
        date = self.datas[0].datetime.date(0)
        filtered = self.df[self.df["date"] == date]
        if len(filtered) == 0:
            return
        actual_signal = filtered.iloc[0]["Signal"]
        if pd.isnull(actual_signal):
            return
        if actual_signal.lower() == "buy" and position == 0:
            self.buy(size=400)
        elif actual_signal.lower() == "sell" and position > 0:
            self.sell(size=position)


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(MySimpleStrategy, signal_file='data2.csv', security="AMD")

    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    data = bt.feeds.Quandl(
        dataname='AMD',
        apikey='XH28RzhxDVHKWwnaN1Hv',
        fromdate=datetime(2017, 1, 1),
        todate=last_date)
    cerebro.adddata(data)

    cerebro.addwriter(bt.WriterFile, csv=True, out='your_strategy_results.csv')

    # Analyzer
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')

    thestrats = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    thestrat = thestrats[0]
    print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis())

    cerebro.plot()

