# Linear Regression using Sci-kit Learn:

import requests
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=BHARTIARTL.BSE&outputsize=full&datatype=csv&apikey=...")
data = response.text
df = pd.read_csv(StringIO(data))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values(by='timestamp', inplace=True)

X = df[['open','high','low']].values
y = df['close'].values
open = X[:,0]
high = X[:,1]
low = X[:,2]
close = y
price_range = high - low
price_spread = close - open
oh_price_spread = high - open
ol_price_spread = open - low
price_change_pct_1 = (price_spread/open) * 100
price_change_pct_2 = ((high - low)/low) * 100
average_price_1 = (open + high + low + close)/4
average_price_2 = (open + high + low)/3
high_volatility = df['high'].rolling(window=20).std()
#high_volatility = high_volatility[227:,]
low_volatility = df['low'].rolling(window=20).std()
#low_volatility = low_volatility[227:,]
oh_increase = np.where(high > open, 1, 0)
ol_decrease = np.where(low < open, 1, 0)
open_position = (open - low) / (high - low)

#Relative_Strenth_Index
def rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
relative_strength_index = rsi(df)

#Monthly_MovingAverage
df.set_index('timestamp', inplace=True)
m_ma = df['close'].resample('ME').mean().rolling(window=3).mean()
m_ma = m_ma.values

#Volatility
monthly_returns = df['close'].pct_change().fillna(0)
monthly_volatility = monthly_returns.std()
annualized_volatility = monthly_volatility * np.sqrt(12)
historical_volatility = monthly_returns.rolling(window=12).std()
monthly_returns = monthly_returns.values
historical_volatility = historical_volatility.values

x1 = price_range
x2 = oh_price_spread
x3 = ol_price_spread
x4 = price_change_pct_2
x5 = average_price_2
x6 = high_volatility
x7 = low_volatility
x8 = oh_increase
x9 = ol_decrease
x10 = open_position
X = np.column_stack((X,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10))

x_train, y_train = X[0:136,:], y[0:136]
x_cv, x_test, y_cv, y_test = X[136:181,:], X[181:227,:], y[136:181], y[181:227]
x_train = x_train[20:,:]
y_train = y_train[20:,]

scaler = StandardScaler()
model = LinearRegression()
poly = PolynomialFeatures(degree=1, include_bias=False)

train_MSEs = []
cv_MSEs = []
scalers = []
models = []
polys = []
deg=11

x_train_mapped = poly.fit_transform(x_train)
x_train_mapped_scaled = scaler.fit_transform(x_train_mapped)
model.fit(x_train_mapped_scaled, y_train)
yhat_train = model.predict(x_train_mapped_scaled)
train_MSE = mean_squared_error(y_train, yhat_train) / 2
print(f"Training Error: {train_MSE}")

x_cv_mapped = poly.transform(x_cv)
x_cv_mapped_scaled = scaler.transform(x_cv_mapped)
yhat_cv = model.predict(x_cv_mapped_scaled)
cv_MSE = mean_squared_error(y_cv, yhat_cv) / 2
print(f"Cross-Validation Error: {cv_MSE}")

m = x_train_mapped_scaled.shape[0]
n = x_train_mapped_scaled.shape[1]
period = np.arange(1,m+1)
print(f"Prediction: {yhat_train}")

plt.scatter(period, y_train, marker='X', c='r', label='Fact')
plt.plot(period, yhat_train, c='b', label='Prediction')
plt.xlabel('No. of months (2005-2023)')
plt.ylabel('Price (in Rs.)')
plt.title(f"Actual vs. Predicted Prices of Bhartiartl in BSE with {n} features with 'Sci-kit Learn'")
plt.legend()
plt.show()
