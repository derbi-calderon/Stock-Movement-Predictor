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
    df['MA_5'] = ta.MA(df['Low'].values, timeperiod=5, matype=0)

    # Add the twenty day moving average technical indicator
    df['MA_20'] = ta.MA(df['Low'].values, timeperiod=20, matype=0)

    # Add the fifty day moving average technical indicator
    df['MA_50'] = ta.MA(df['Low'].values, timeperiod=50, matype=0)

    # Add the Bollinger Bands technical indicators
    df['BOL_Upp'], df['BOL_Mid'], df['BOL_Low'] = ta.BBANDS(df['Low'].values,
                                                            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # Add the Relative strength index technical indicator
    df['RSI'] = ta.RSI(df['Low'].values, 14)

    # Add the Simple Moving Average (Fast & Slow) technical indicators
    df['SMA_Fast'] = ta.SMA(df['Low'].values, 5)
    df['SMA_Slow'] = ta.SMA(df['Low'].values, 20)

    # Add the percent change of the daily closing price
    df['ClosingPctChange'] = df['Low'].pct_change()

    # Get today's record (the last record) so we can predict it later. Do this
    # before we add the 'NextDayPrice' column so we don't have to drop it later
    df_today = df.iloc[-1:, :].copy()

    # Create a column of the next day's closing prices so we can train on it
    # and then eventually predict the value
    df['NextClose'] = df['Low'].shift(-1)

    # Get rid of the rows that have NaNs
    df.dropna(inplace=True)

    # Decide which features to use for our regression. This will allow us to
    # tweak things during testing
    features_to_fit = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50',
                       'RSI', 'SMA_Fast', 'SMA_Slow', 'BOL_Upp', 'BOL_Mid', 'BOL_Low', 'ClosingPctChange']

    # Create our target and labels
    X = df[features_to_fit]
    y = df['NextClose']

    # Create training and testing data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=42)

    # Do ten-fold cross-validation and compute our average accuracy
    cv = cross_val_score(estimator, X_test, y_test, cv=10)
    print('Accuracy:', cv.mean())

    # Fit the regressor with the full dataset to be used with predictions
    estimator.fit(X, y)

    # Predict today's closing price
    X_new = df_today[features_to_fit]
    next_price_prediction = estimator.predict(X_new)

    # Return the predicted closing price
    return next_price_prediction


# Choose which company to predict
symbol = ['ADP', 'SPY', 'BA', 'KSU', 'GME', 'MJ', 'LODE', 'CLF', 'TMUS', 'AXP', 'ABBV', 'KR', 'VZ' ,
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


for sym in symbol:
    print("Prediction for  ---" + sym)
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

    print(r_df)
    quotes_df = r_df

    # Predict the last day's closing price using linear regression
    print('Unscaled Linear Regression:')
    linreg = LinearRegression()
    print('Predicted Closing Price: %.2f\n' % make_prediction(quotes_df, linreg))

    # Predict the last day's closing price using Linear regression with scaled features
    print('Scaled Linear Regression:')
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    print('Predicted Closing Price: %.2f\n' % make_prediction(quotes_df, pipe))

    # Predict the last day's closing price using ridge regression
    print('Unscaled Ridge Regression:')
    ridge = Ridge(normalize=True)
    print('Predicted Closing Price: %.2f\n' % make_prediction(quotes_df, ridge))

    # Predict the last day's closing price using ridge regression and scaled features
    print('Scaled Linear Regression:')
    ridge_pipe = make_pipeline(StandardScaler(), Ridge())
    print('Predicted Closing Price: %.2f\n' % make_prediction(quotes_df, ridge_pipe))

    # Predict the last day's closing price using decision tree regression
    print('Unscaled Decision Tree Regressor:')
    tree = DecisionTreeRegressor()
    print('Predicted Closing Price: %.2f\n' % make_prediction(quotes_df, tree))

    '''
    # Predict the last day's closing price using Gaussian Naive Bayes
    print('Unscaled Gaussian Naive Bayes:')
    nb = GaussianNB()
    print('Predicted Closing Price: %.2f\n' % make_prediction(quotes_df, nb))
    '''