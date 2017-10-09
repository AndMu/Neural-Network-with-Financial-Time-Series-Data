import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

from DataProcessing import DataProcessing
from MarketData import MarketData
from PlotManager import PlotManager
from PredictionModel import PredictionModelFactory

sns.set()
import quandl

quandl.ApiConfig.api_key = 'XH28RzhxDVHKWwnaN1Hv'
seq_len = 22

predict_days = 10
plot_days = 30
stock_name = 'AMD'
load = False
weight_file = stock_name + '_trained_reg.h5'

data = MarketData(stock_name, ma=[50, 100, 200])
df = data.get_stock_data()
df = df.loc[df.index < '2017-01-01', :].copy()
# plot_stock(df)
#
# corr = df.corr()
# ax = sns.heatmap(corr, cmap="YlGnBu")
# plt.show()

x_data, y_data_originl = DataProcessing.load_data(df, seq_len)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_originl, test_size=0.2)

model = PredictionModelFactory.create_default(seq_len)

if load:
    model.model.load_weights(weight_file)
else:
    model.fit(x_train, y_train)
    model.model.save_weights(weight_file)
    model.model_score(x_train, y_train, x_test, y_test)
    p = model.percentage_difference(x_test, y_test)

prices = model.predict_days(data, predict_days)
print(prices)

x_data, y_data_final = DataProcessing.load_data(df, seq_len)
PlotManager.plot_result(data, y_data_final[(-predict_days - plot_days + 1):-1], y_data_originl[-plot_days:-1])