from __future__ import print_function

import numpy as np
import pandas as pd
import talib as ta
import pandas_datareader as web

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def make_prediction(quotes_df, estimator):
    # Make a copy of the dataframe so we don't modify the original
    df = quotes_df.copy()

    # Add the five day moving average technical indicator
    df['MA_5'] = ta.MA(df['Close'].values, timeperiod=15, matype=0)
    df['MA_5_o'] = ta.MA(df['Open'].values, timeperiod=15, matype=0)

    # Add the twenty day moving average technical indicator
    df['MA_20'] = ta.MA(df['Close'].values, timeperiod=30, matype=0)
    df['MA_20_o'] = ta.MA(df['Open'].values, timeperiod=30, matype=0)

    # Add the Bollinger Bands technical indicators
    df['BOL_Upp'], df['BOL_Mid'], df['BOL_Low'] = ta.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2,
                                                            nbdevdn=2, matype=0)

    df['BOL_Upp_o'], df['BOL_Mid_o'], df['BOL_Low_o'] = ta.BBANDS(df['Open'].values, timeperiod=20, nbdevup=2,
                                                                  nbdevdn=2, matype=0)


    # Add the Relative strength index technical indicator
    df['RSI'] = ta.RSI(df['Close'].values, 14)
    df['RSI_o'] = ta.RSI(df['Open'].values, 14)

    # Add the Simple Moving Average (Fast & Slow) technical indicators
    df['SMA_Fast'] = ta.SMA(df['Close'].values, 6)
    df['SMA_Slow'] = ta.SMA(df['Close'].values, 21)

    df['SMA_Fast_o'] = ta.SMA(df['Open'].values, 6)
    df['SMA_Slow_o'] = ta.SMA(df['Open'].values, 21)



    # add ons
    df['MOM'] = ta.MOM(df['Close'].values, timeperiod=5)
    df['MOM_o'] = ta.MOM(df['Open'].values, timeperiod=5)

    df['PlUS_DI_Fast'] = ta.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, 6)
    df['PlUS_DI_Slow'] = ta.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, 12)

    df['AD'] = ta.AD(df['High'], df['Low'], df['Close'], df['Volume'])

    # linear regress indicators
    df['LINEARREG_7'] = ta.LINEARREG(df['Close'].values, 7)
    df['LINEARREG_14'] = ta.LINEARREG(df['Close'].values, 14)

    df['LINEARREG_7_o'] = ta.LINEARREG(df['Open'].values, 7)
    df['LINEARREG_14_o'] = ta.LINEARREG(df['Open'].values, 14)

    df['LINEARREG_ANGLE_7'] = ta.LINEARREG_ANGLE(df['Close'].values, 7)
    df['LINEARREG_ANGLE_14'] = ta.LINEARREG_ANGLE(df['Close'].values, 14)

    df['LINEARREG_ANGLE_7_o'] = ta.LINEARREG_ANGLE(df['Open'].values, 7)
    df['LINEARREG_ANGLE_14_o'] = ta.LINEARREG_ANGLE(df['Open'].values, 14)

    df['LINEARREG_INTERCEPT_7'] = ta.LINEARREG_INTERCEPT(df['Close'].values, 7)
    df['LINEARREG_INTERCEPT_14'] = ta.LINEARREG_INTERCEPT(df['Close'].values, 14)

    df['LINEARREG_INTERCEPT_7_o'] = ta.LINEARREG_INTERCEPT(df['Open'].values, 7)
    df['LINEARREG_INTERCEPT_14_o'] = ta.LINEARREG_INTERCEPT(df['Open'].values, 14)

    df['LINEARREG_SLOPE_7'] = ta.LINEARREG_SLOPE(df['Close'].values, 7)
    df['LINEARREG_SLOPE_14'] = ta.stream_LINEARREG_SLOPE(df['Close'].values, 14)

    df['LINEARREG_SLOPE_7_o'] = ta.LINEARREG_SLOPE(df['Open'].values, 7)
    df['LINEARREG_SLOPE_14_o'] = ta.stream_LINEARREG_SLOPE(df['Open'].values, 14)

    df['MEDPRICE'] = ta.MEDPRICE(df['High'], df['Low'])

    # Add the percent change of the daily closing price
    df['ClosingPctChange'] = df['Close'].pct_change()

    # Get today's record (the last record) so we can predict it later. Do this
    # before we add the 'NextDayPrice' column so we don't have to drop it later
    df_today = df.iloc[-1:, :].copy()

    # Create a column of the next day's closing prices so we can train on it
    # and then eventually predict the value
    df['NextClose'] = df['Close'].shift(-1)

    # Get rid of the rows that have NaNs
    df.dropna(inplace=True)

    # Decide which features to use for our regression. This will allow us to
    # tweak things during testing
    features_to_fit = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20',
                       'RSI', 'SMA_Fast', 'SMA_Slow', 'BOL_Upp', 'BOL_Mid', 'BOL_Low', 'ClosingPctChange',
                       'MOM', 'PlUS_DI_Fast', 'PlUS_DI_Slow', 'LINEARREG_7', 'LINEARREG_14',
                       'MOM', 'PlUS_DI_Fast', 'PlUS_DI_Slow', 'LINEARREG_7', 'LINEARREG_14',
                       'AD', 'LINEARREG_SLOPE_14', 'LINEARREG_SLOPE_7', 'LINEARREG_ANGLE_7', 'LINEARREG_ANGLE_14',
                       'LINEARREG_INTERCEPT_7', 'LINEARREG_INTERCEPT_14',
                       'MEDPRICE', 'MA_5_o', 'MA_20_o', 'BOL_Upp_o', 'BOL_Mid_o', 'BOL_Low_o', 'RSI_o', 'SMA_Fast_o',
                       'SMA_Slow_o', 'MOM_o','LINEARREG_7_o', 'LINEARREG_14_o', 'LINEARREG_ANGLE_7',
                       'LINEARREG_ANGLE_14', 'LINEARREG_INTERCEPT_7_o','LINEARREG_INTERCEPT_14_o',
                       'LINEARREG_SLOPE_7_o', 'LINEARREG_SLOPE_14_o']

    # Create our target and labels
    X = df[features_to_fit]
    y = df['NextClose']

    # Create training and testing data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,
                                                        random_state=42)

    # Do ten-fold cross-validation and compute our average accuracy
    cv = cross_val_score(estimator, X_test, y_test, cv=10)
    #  print('Accuracy:', cv.mean())

    # Fit the regressor with the full dataset to be used with predictions

    estimator.fit(X, y)

    # Predict today's closing price
    X_new = df_today[features_to_fit]
    next_price_prediction = estimator.predict(X_new)

    # Return the predicted closing price
    return next_price_prediction


# Choose which company to predict
symbol = ['SPY', 'ADP', 'BA', 'KSU', 'GME', 'MJ', 'LODE', 'CLF', 'TMUS', 'AXP', 'ABBV', 'KR', 'VZ' ,
           'MMC', 'CVX', 'MRK', 'BMY', 'DKNG', 'RIOT', 'TKAT', 'CLVS', 'BIDU', 'VIAC',
          'FNKO', 'PRQR', 'MU', 'MARA', 'TSLA', 'AMC', 'AAPL'
          , 'GOLD', 'GM', 'CVX', 'KR', 'PFE', 'BMY', 'PFE', 'USB', 'RH', 'WFC', 'V', 'MSFT', 'AMZN', 'FB'
        , 'GOOG', 'JPM', 'UNH', 'NVDA', 'DIS', 'HD', 'PG'
        ,'MA', 'BAC', 'PYPL', 'INTC', 'CMCSA', 'NFLX', 'XOM'
        , 'ADBE', 'CSCO', 'T', 'ABT', 'KO', 'CRM', 'CVX'
        , 'AVGO', 'PEP', 'WMT', 'TMO', 'TXN', 'ACN', 'NKE', 'MCD',
        'MDT', 'COST', 'QCOM', 'HON', 'C', 'NEE', 'UNP', 'LIN', 'LLY',
        'AMGN', 'DHR', 'LOW',  'BMY', 'PM', 'ORCL', 'AMAT', 'SBUX',
        'CAT', 'UPS', 'IBM', 'RTX', 'GE', 'DE', 'MS',
        'MMM', 'GS', 'BLK', 'INTU', 'AMT', 'MU', 'TGT'
        ,'SCHW', 'NOW', 'BKNG', 'AMD', 'CVS', 'MO','AXP', 'LRCX', 'LMT', 'FIS', 'ISRG', 'SPGI'
        , 'ANTM', 'CHTR', 'CI', 'GILD', 'MDLZ'
        ,'ADP', 'TJX', 'SYK', 'PLD', 'TFC', 'USB', 'CCI', 'ATVI'
        , 'PNC', 'ZTS', 'CSX', 'DUK']

'''
import datetime

start = datetime.datetime(20, 1, 1)

end = datetime.datetime(2021, 4, 8)
'''
nm = 0
for l in symbol:
    nm += 1

nr = 0
for sym in symbol:
    # print(sym) # for testing
    avgprice = 0
    completed = (nr / nm) * 100
   # if completed % 5 == 0:
     #   print(str(completed) + "%")
    nr += 1
    # Import a year's OHLCV data from yahoo using DataReader
    quotes_df = web.data.DataReader(sym, 'yahoo')
    agg_dict = {'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Adj Close': 'last',
                'Volume': 'mean'}

    # resampled dataframe
    # 'W' means weekly aggregation
    r_df = quotes_df.resample('W').agg(agg_dict)
    quotes_df = r_df
    # print(quotes_df)

    a = quotes_df['Close']
    # print(a)
    if len(quotes_df['High']) < 5:
        continue
    aa = a[-1]
    found = False
    signal = 4.20
    # Predict the last day's closing price using linear regression

    linreg = LinearRegression()
    ln = make_prediction(quotes_df, linreg)
    avgprice += ln
    if (aa - signal) > ln or ln > (aa + signal):
        print("Prediction for  ---" + sym)
        found = True
        print('Unscaled Linear Regression:')
        print(' Strong Buy / SELL  = Predicted Closing Price: %.2f\n' % ln)
        print('Last Close:' + str(aa))

    # Predict the last day's closing price using Linear regression with scaled features

    pipe = make_pipeline(StandardScaler(), LinearRegression())
    p = make_prediction(quotes_df, pipe)
    avgprice += p
    if (aa - signal) > p or p > (aa + signal):
        print("Prediction for  ---" + sym)
        found = True
        print('Scaled Linear Regression:')
        print(' Strong Buy / SELL  = Predicted Closing Price: %.2f\n' % p)
        print('Last Close:' + str(aa))

    # Predict the last day's closing price using ridge regression

    ridge = Ridge(normalize=True)
    r = make_prediction(quotes_df, ridge)
    avgprice += r
    if (aa - 10) > r or r > (aa + 10):
        print("Prediction for  ---" + sym)
        found = True
        print('Unscaled Ridge Regression:')
        print(' Strong Buy / SELL  = Predicted Closing Price: %.2f\n' % r)
        print('Last Close:' + str(aa))

    # Predict the last day's closing price using ridge regression and scaled features

    ridge_pipe = make_pipeline(StandardScaler(), Ridge())
    rr = make_prediction(quotes_df, ridge_pipe)
    avgprice += rr
    if (aa - signal) > rr or rr > (aa + signal):
        print("Prediction for  ---" + sym)
        found = True
        print('Scaled Linear Regression:')
        print(' Strong Buy / SELL  = Predicted Closing Price: %.2f\n' % rr)
        print('Last Close:' + str(aa))

    # Predict the last day's closing price using decision tree regression

    tree = DecisionTreeRegressor()
    t = make_prediction(quotes_df, tree)
    avgprice += t
    if (aa - signal) > t or t > (aa + signal):
        print("Prediction for  ---" + sym)
        found = True
        print('Unscaled Decision Tree Regressor:')
        print(' Strong Buy / SELL  = Predicted Closing Price: %.2f\n' % t)
        print('Last Close:' + str(aa))

    if found == True:
        avgprice /= 5
        print("Average Price: %.2f\n" % avgprice)

print("end")
