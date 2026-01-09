
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv('Stock Prices Data Set.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)

df['close'].plot(title="Stock Price Trend")
plt.show()

decomposition = seasonal_decompose(df['close'], model='additive', period=30)
decomposition.plot()
plt.show()
