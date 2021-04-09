from __future__ import print_function
import pandas_datareader as web
# import datetime
import Stock_Names as SN
import make_prediction as mp

def
# Choose which company to predict
tickers = SN.st_names()

'''
# Choose dates
start = datetime.datetime(20, 1, 1)

end = datetime.datetime(2021, 4, 8)
'''

for tick in tickers:

    print("Prediction for  ---" + tick)
    weekly = False                                      # True - weekly data | False - daily data
    if weekly:
        quotes_df = web.data.DataReader(tick, 'yahoo')  # Import  OHLCV data from yahoo finance using DataReader
        agg_dict = {'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Adj Close': 'last',
                    'Volume': 'mean'}
        r_df = quotes_df.resample('W').agg(agg_dict)   # resampled dataframe('W' means weekly aggregation)
        quotes_df = r_df
        print(quotes_df)
    else:
        quotes_df = web.data.DataReader(tick, 'yahoo')  # Import OHLCV data from yahoo finance using DataReader
        print(quotes_df)