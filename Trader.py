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

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])
        # self.sell()
        position = self.getpositionbyname(self.params.security)
        date = self.datas[0].datetime.date(0)
        filtered = self.df[self.df["date"] == date]
        if len(filtered) == 0:
            return
        hold = filtered.iloc[0]["Result-0"]
        buy = filtered.iloc[0]["Result-1"]
        sell = filtered.iloc[0]["Result-2"]

        if buy > 0.5 and position.size == 0:
            self.log('BUY CREATE, %.2f' % self.data.close[0])
            self.buy(price=self.data.close[0], size=400)
        elif sell > 0.5 and position.size > 0:
            self.log('SELL CREATE, %.2f' % self.data.close[0])
            self.sell(price=self.data.close[0], size=position.size)


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(MySimpleStrategy, signal_file='result.csv', security="AMD")

    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.broker.setcommission(commission=0.001)

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

