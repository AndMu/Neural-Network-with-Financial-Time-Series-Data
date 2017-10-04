import seaborn as sns;

from DataProcessing import DataProcessing
from MarketData import MarketData
from PlotManager import PlotManager
from PredictionModel import PredictionModel

sns.set()
import quandl

quandl.ApiConfig.api_key = 'XH28RzhxDVHKWwnaN1Hv'
seq_len = 22
shape = [seq_len, 9, 1]
neurons = [256, 256, 32, 1]
dropout = 0.3
decay = 0.5
epochs = 90
stock_name = 'AMD'


df = MarketData.get_stock_data(stock_name, ma=[50, 100, 200])
# plot_stock(df)
#
# corr = df.corr()
# ax = sns.heatmap(corr, cmap="YlGnBu")
# plt.show()

X_train, y_train, X_test, y_test = DataProcessing.load_data(df, seq_len)
model = PredictionModel(shape, neurons, dropout, decay)

model.fit(X_train, y_train)

model.model_score(X_train, y_train, X_test, y_test)

p = model.percentage_difference(X_test, y_test)
PlotManager.plot_result(stock_name, p, y_test)