import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
df = pd.read_csv('newdata.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].apply(mdates.date2num)
fig, ax = plt.subplots()
candlestick_ohlc(ax, df.values, width=0.6, colorup='g', colordown='r')
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("Ethereum Price")
plt.show()
