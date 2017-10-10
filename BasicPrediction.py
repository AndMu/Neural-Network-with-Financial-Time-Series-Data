import seaborn as sns
from sklearn.model_selection import train_test_split

from AnalysisPackage.Indicators import CombinedIndicator, MomentumIndicator, BollingerIndicator, RsiIndicator
from DataProcessing import DataProcessing
from MarketData import MarketData, MarketDataSource
from PlotManager import PlotManager
from PredictionModel import PredictionModelFactory

sns.set()
import quandl

quandl.ApiConfig.api_key = 'XH28RzhxDVHKWwnaN1Hv'
seq_len = 22

predict_days = 5
plot_days = 30
stock_name = 'TSLA'
load = False
weight_file = stock_name + '_trained_reg.h5'
df = MarketDataSource().get_stock_data_basic(stock_name)
df = df.loc[df.index < '2017-01-01', :].copy()
indicators = CombinedIndicator((MomentumIndicator(), BollingerIndicator(), RsiIndicator()))
data = MarketData(stock_name, df, ma=[50, 100, 200], indicator=indicators)
# plot_stock(df)
#
# corr = df.corr()
# ax = sns.heatmap(corr, cmap="YlGnBu")
# plt.show()

df = data.get_stock_data()
x_data, y_data_original = DataProcessing.load_data(df, seq_len)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_original, test_size=0.2)

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

df = data.get_stock_data()
x_data, y_data_final = DataProcessing.load_data(df, seq_len)
PlotManager.plot_result(data, y_data_final[(-predict_days - plot_days):-1], y_data_original[-plot_days:-1])