import abc
import logging

import pandas as pd
import talib

from Analysis import TechnicalPerformance
from Definitions import HeaderFactory

logger = logging.getLogger(__name__)


class Indicator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def calculate(self, data):
        pass

    @abc.abstractmethod
    def required_days(self) :
        pass


class CombinedIndicator(Indicator):

    def required_days(self):
        return max(self.indicators, key=lambda x: x.required_days()).required_days()

    def __init__(self, indicators):
        self.indicators = indicators

    def calculate(self, data):
        result = pd.DataFrame(index=data.index)
        for indicator in self.indicators:
            indicator_result = indicator.calculate(data)
            result = result.join(indicator_result)
        return result


class MomentumIndicator(Indicator):

    def __init__(self, days=5):
        self.days = days

    def required_days(self):
        return self.days + 1

    def calculate(self, data):
        data = data[HeaderFactory.Price].copy()
        previous = data.shift(self.days)
        data = data / previous - 1
        result = pd.DataFrame(index=data.index, data=data.values, columns=[HeaderFactory.MOM])
        return result


class BollingerIndicator(Indicator):

    def __init__(self, windows=20):
        self.windows = windows

    def required_days(self):
        return self.windows

    def calculate(self, data):
        data = data[HeaderFactory.Price].copy()
        rm, rstd = TechnicalPerformance.compute_std(data, self.windows)
        result = (data - rm) / (2 * rstd)
        result = pd.DataFrame(index=result.index, data=result.values, columns=[HeaderFactory.Bollinger])
        return result


class RsiIndicator(Indicator):

    def required_days(self):
        return 15

    def calculate(self, data):
        data = pd.DataFrame(data[HeaderFactory.Price])
        rsi = data.apply(RsiIndicator._RSI, axis=0)
        rsi /= 100
        data = rsi[HeaderFactory.Price]
        result = pd.DataFrame(index=data.index, data=data.values, columns=[HeaderFactory.RSI])
        return result

    @staticmethod
    def _RSI(prices):
        rsi = talib.RSI(prices.values)
        return rsi


class MACDIndicator(Indicator):

    def __init__(self):
        self.n_fast = 12
        self.n_slow = 26
        self.signal_period = 9

    def required_days(self):
        return 26

    def calculate(self, data, normalized=False):
        fast = data[HeaderFactory.Price].ewm(adjust=True, min_periods=self.n_slow - 1, span=self.n_fast,
                                             ignore_na=False).mean()
        EMAfast = pd.Series(fast)

        slow = data[HeaderFactory.Price].ewm(adjust=True, min_periods=self.n_slow - 1, span=self.n_slow,
                                             ignore_na=False).mean()
        EMAslow = pd.Series(slow)
        result = EMAfast - EMAslow
        if normalized:
            result = result / EMAslow
        MACD = pd.Series(result, name=HeaderFactory.MACD)
        signal = MACD.ewm(adjust=True, min_periods=self.signal_period - 1, span=self.signal_period,
                          ignore_na=False).mean()
        MACDsign = pd.Series(signal, name=HeaderFactory.MACD_SIGNAL)
        MACDdiff = pd.Series(MACD - MACDsign, name=HeaderFactory.MACD_DIFF)
        data = pd.DataFrame(MACD).join(MACDsign)
        data = data.join(MACDdiff)
        return data

