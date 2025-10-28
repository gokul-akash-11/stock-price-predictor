# Linear Regression - Manual Implementation:

import requests
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

#response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=BHARTIARTL.BSE&datatype=csv&apikey=...")
#data = response.text

df = pd.read_csv('monthly_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values(by='timestamp', inplace=True)

x_train = df[['open','high','low']].values
y_train = df['close'].values

open = x_train[:,0]
high = x_train[:,1]
low = x_train[:,2]
close = y_train
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
m_ma = df['close'].ewm(span=20).mean()

#Volatility
monthly_returns = df['close'].pct_change().fillna(0)
monthly_volatility = monthly_returns.std()
annualized_volatility = monthly_volatility * np.sqrt(12)
historical_volatility = monthly_returns.rolling(window=12).std()
monthly_returns = monthly_returns.values
historical_volatility = historical_volatility.values


x_train = np.column_stack((x_train, price_range, oh_price_spread, ol_price_spread, price_change_pct_2, average_price_2, high_volatility, low_volatility, oh_increase, ol_decrease, open_position))
x_train = x_train[20:,:]
y_train = y_train[20:,]
w_ini = np.ones(x_train.shape[1])
b_ini = 0
alpha = 1e-6
lambda_ = 1
m, n = x_train.shape[0], x_train.shape[1]
period = np.arange(1,m+1)
print(m,n)

def main():
    w, b, loss = gradient_descent(x_train,y_train,w_ini,b_ini,alpha,lambda_,compute_cost,compute_gradient,10000)
    f_wb = np.zeros(m, dtype=float)
    for i in range(m):
        f_wb[i] = np.dot(w, x_train[i]) + b
    print(f"w final: {w}\nb final: {b}")
    print("f_wb:", f_wb)
    specifications = f"Training eg.s = {m}\nFeatures = {n}\nAlpha = {alpha}\nLoss = {loss:.2f}"

    plt.scatter(period, y_train, marker='X', c='r', label='Fact')
    plt.plot(period, f_wb, c='b', label='Prediction')
    plt.xlabel('No. of Days (Starting from 3-1-2005)')
    plt.ylabel('Price[INR]')
    plt.title(f'''Actual vs. Predicted Prices of Bhartiartl in BSE with {n} features''')
    plt.legend(loc='upper left')
    plt.text(0.81,0.86,specifications,transform=plt.gca().transAxes,bbox=dict(facecolor='white',alpha=0.8),fontsize=10)
    plt.show()

def compute_cost(x,y,w,b,lambda_):
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(w, x[i]) + b
        err_sq = (f_wb_i - y[i]) ** 2
        cost += err_sq
    cost /= 2*m
    
    reg = 0
    for j in range(n):
        reg += w[j]**2
    #cost += (lambda_/(2*m)) * reg   # Regularization
    return cost

def compute_gradient(x,y,w,b,lambda_):
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        f_wb_i = np.dot(w, x[i]) + b
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    
    reg = 0
    for j in range(n):
        reg += w[j]
    #dj_dw += (lambda_/m) * reg   # Regularization
    return dj_dw, dj_db

def gradient_descent(x,y,w,b,alpha,lambda_,cost_function,gradient_function,iters):
    j_hist = []
    a=0
    for i in range(iters+1):
        if i % 1000 == 0:
            j_hist.append(cost_function(x,y,w,b,lambda_))
            print(f"{i}th iteration: {j_hist[a]}")
            a+=1
        dj_dw, dj_db = gradient_function(x,y,w,b,lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
    return w, b, j_hist[a-1]

if __name__ == "__main__":
    main()
