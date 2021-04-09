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
# Choose which company to predict
symbol = ['CTRM']


for sym in symbol:
    print("Prediction for  ---" + sym)
    # Import a year's OHLCV data from yahoo using DataReader

    logic = {'Open': 'first',
             'High': 'max',
             'Low': 'min',
             'Close': 'last',
             'Volume': 'sum'}
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
