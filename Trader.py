from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import datetime, date, timedelta

import backtrader as bt
import backtrader.analyzers as btanalyzers

from AnalysisPackage.Indicators import CombinedIndicator, MomentumIndicator, BollingerIndicator, RsiIndicator
from MarketData import MarketDataSource, MarketData
from PredictionModel import PredictionModelFactory

daysOffset = 10
last_date = date.today()
# last_date = date.today() - + timedelta(days=daysOffset)
seq_len = 21


class MySimpleStrategy(bt.Strategy):

    params = (
        ('security', ''),
    )

    def __init__(self):
        self.model = PredictionModelFactory.create_default(seq_len)
        self.all_prices = MarketDataSource().get_stock_data_basic(self.params.security)
        weight_file = self.params.security + '_trained_reg.h5'
        self.model.model.load_weights(weight_file)

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
        date = datetime.combine(date, datetime.min.time())
        filtered = self.all_prices[self.all_prices.index <= date].copy()
        indicators = CombinedIndicator((MomentumIndicator(), BollingerIndicator(), RsiIndicator()))
        market_data = MarketData(self.params.security, filtered, ma=[50, 100, 200], indicator=indicators)
        total_days = 5
        prices = self.model.predict_days(market_data, total_days)
        last_price = prices[-1]

        if last_price > (self.data.close[0] * 1.01) and position.size == 0:
            self.log('BUY CREATE, %.2f' % self.data.close[0])
            self.buy(price=self.data.close[0], size=800)
        elif (last_price * 1.01) <= self.data.close[0] and position.size > 0:
            self.log('SELL CREATE, %.2f' % self.data.close[0])
            self.sell(price=self.data.close[0], size=position.size)


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    stock = "TSLA"
    # Add a strategy
    cerebro.addstrategy(MySimpleStrategy, security=stock)

    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.broker.setcommission(commission=0.001)

    data = bt.feeds.Quandl(
        dataname=stock,
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

