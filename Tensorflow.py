# Tensorflow Implementation of Regression Problem:

import requests
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

#response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=BHARTIARTL.BSE&outputsize=full&datatype=csv&apikey=...")
#data = response.text

df = pd.read_csv('daily_data.csv')
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
low_volatility = df['low'].rolling(window=20).std()
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

X = np.column_stack((X, price_range, oh_price_spread, ol_price_spread, price_change_pct_2, average_price_2, high_volatility, low_volatility, oh_increase, ol_decrease, open_position))
X = X[20:,:]
y = y[20:,]

#Split the Dataset
x_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
x_dev, x_test, y_dev, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
del x_, y_

#Mapping Features
mapp = PolynomialFeatures(1, include_bias=False)
x_train_mapped = mapp.fit_transform(x_train)
x_dev_mapped = mapp.transform(x_dev)
x_test_mapped = mapp.transform(x_test)

#Scaling Features
scale = StandardScaler()
x_train_mapped_scaled = scale.fit_transform(x_train_mapped)
x_dev_mapped_scaled = scale.transform(x_dev_mapped)
x_test_mapped_scaled = scale.transform(x_test_mapped)

X_mapped = mapp.fit_transform(X)
X_mapped_scaled = scale.fit_transform(X_mapped)
m, n = X_mapped_scaled.shape[0], X_mapped_scaled.shape[1]
period = np.arange(1, m+1)


model = Sequential(
                   [Dense(50, activation='relu', name='l1', input_shape=(n,)),
                    Dense(25, activation='relu', name='l2'),
                    Dense(10, activation='relu', name='l3'),
                    Dense(1, activation='linear', name='l4'),
                    ]
                )
alpha = 0.0004
model.compile(
                optimizer=Adam(learning_rate=alpha),
                loss=MeanSquaredError()
            )
epochs = 100
history = model.fit(X_mapped_scaled, y, epochs=epochs)
loss = history.history['loss'][-1]
yhat = model.predict(X_mapped_scaled)
specifications = f"Training eg.s = {m}\nFeatures = {n}\nLayers = 4\nUnits = 50,25,10,1\nAlpha = {alpha}\nEpochs = {epochs}\nLoss = {loss:.2f}"

plt.scatter(period, y, marker='X', c='r', label='Fact')
plt.plot(period, yhat, c='b', label='Prediction')
plt.xlabel('No. of Days (from 3-1-2005 to 16-1-2024)')
plt.ylabel('Price[INR]')
plt.title(f'''Actual vs. Predicted Prices of Bhartiartl in BSE, using "Neural Networks"''')
plt.legend(loc='upper left')
plt.text(0.81,0.78,specifications,transform=plt.gca().transAxes,bbox=dict(facecolor='white',alpha=0.8),fontsize=10)
plt.show()
