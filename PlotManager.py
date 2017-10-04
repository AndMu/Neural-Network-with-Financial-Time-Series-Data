import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from MarketData import MarketData


class PlotManager(object):

    @staticmethod
    def plot_stock(df):
        print(df.head())
        plt.subplot(211)
        plt.plot(df['Adj Close'], color='red', label='Adj Close')
        plt.legend(loc='best')
        plt.subplot(212)
        plt.plot(df['Pct'], color='blue', label='Percentage change')
        plt.legend(loc='best')
        plt.show()

    @staticmethod
    def plot_result(stock_name, normalized_value_p, normalized_value_y_test):
        newp = MarketData.denormalize(stock_name, normalized_value_p)
        newy_test = MarketData.denormalize(stock_name, normalized_value_y_test)
        print("Last five days price:", newy_test[-5:])
        print("Last five days prediction + future:", newp[-6:])
        plt2.plot(newp, color='red', label='Prediction')
        plt2.plot(newy_test, color='blue', label='Actual')
        plt2.legend(loc='best')
        plt2.title('The test result for {}'.format(stock_name))
        plt2.xlabel('Days')
        plt2.ylabel('Adjusted Close')
        plt2.show()
