from __future__ import print_function
import talib as ta


def technical_indicators(df):
    # Add the five day moving average technical indicator
    df['MA_5'] = ta.MA(df['Close'].values, timeperiod=15, matype=0)
    df['MA_5_o'] = ta.MA(df['Open'].values, timeperiod=15, matype=0)
    df['MA_5_v'] = ta.MA(df['Volume'].values, timeperiod=15, matype=0)

    # Add the twenty day moving average technical indicator
    df['MA_20'] = ta.MA(df['Close'].values, timeperiod=30, matype=0)
    df['MA_20_o'] = ta.MA(df['Open'].values, timeperiod=30, matype=0)
    df['MA_20_v'] = ta.MA(df['Volume'].values, timeperiod=30, matype=0)

    # Add the Bollinger Bands technical indicators
    df['BOL_Upp'], df['BOL_Mid'], df['BOL_Low'] = ta.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2,
                                                            nbdevdn=2, matype=0)

    df['BOL_Upp_o'], df['BOL_Mid_o'], df['BOL_Low_o'] = ta.BBANDS(df['Open'].values, timeperiod=20, nbdevup=2,
                                                                  nbdevdn=2, matype=0)

    df['BOL_Upp_v'], df['BOL_Mid_v'], df['BOL_Low_v'] = ta.BBANDS(df['Volume'].values, timeperiod=20, nbdevup=2,
                                                                  nbdevdn=2, matype=0)
    # Add the Relative strength index technical indicator
    df['RSI'] = ta.RSI(df['Close'].values, 14)
    df['RSI_o'] = ta.RSI(df['Open'].values, 14)

    # Add the Simple Moving Average (Fast & Slow) technical indicators
    df['SMA_Fast'] = ta.SMA(df['Close'].values, 6)
    df['SMA_Slow'] = ta.SMA(df['Close'].values, 21)

    df['SMA_Fast_o'] = ta.SMA(df['Open'].values, 6)
    df['SMA_Slow_o'] = ta.SMA(df['Open'].values, 21)
    'MA_5_v', 'MA_20_v', 'BOL_Upp_v', 'BOL_Mid_v', 'BOL_Low_v'
    # add ons
    df['MOM'] = ta.MOM(df['Close'].values, timeperiod=5)
    df['MOM_o'] = ta.MOM(df['Open'].values, timeperiod=5)

    df['PlUS_DI_Fast'] = ta.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, 6)
    df['PlUS_DI_Slow'] = ta.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, 12)

    df['AD'] = ta.AD(df['High'], df['Low'], df['Close'], df['Volume'])

    # linear regress indicators
    df['LINEARREG_7'] = ta.LINEARREG(df['Close'].values, 7)
    df['LINEARREG_14'] = ta.LINEARREG(df['Close'].values, 14)

    # df['LINEARREG_7_o'] = ta.LINEARREG(df['Open'].values, 7)
    # df['LINEARREG_14_o'] = ta.LINEARREG(df['Open'].values, 14)

    df['LINEARREG_ANGLE_7'] = ta.LINEARREG_ANGLE(df['Close'].values, 7)
    df['LINEARREG_ANGLE_14'] = ta.LINEARREG_ANGLE(df['Close'].values, 14)

    # df['LINEARREG_ANGLE_7_o'] = ta.LINEARREG_ANGLE(df['Open'].values, 7)
    # df['LINEARREG_ANGLE_14_o'] = ta.LINEARREG_ANGLE(df['Open'].values, 14)

    df['LINEARREG_INTERCEPT_7'] = ta.LINEARREG_INTERCEPT(df['Close'].values, 7)
    df['LINEARREG_INTERCEPT_14'] = ta.LINEARREG_INTERCEPT(df['Close'].values, 14)

    # df['LINEARREG_INTERCEPT_7_o'] = ta.LINEARREG_INTERCEPT(df['Open'].values, 7)
    # df['LINEARREG_INTERCEPT_14_o'] = ta.LINEARREG_INTERCEPT(df['Open'].values, 14)

    df['LINEARREG_SLOPE_7'] = ta.LINEARREG_SLOPE(df['Close'].values, 7)
    df['LINEARREG_SLOPE_14'] = ta.stream_LINEARREG_SLOPE(df['Close'].values, 14)

    # df['LINEARREG_SLOPE_7_o'] = ta.LINEARREG_SLOPE(df['Open'].values, 7)
    # df['LINEARREG_SLOPE_14_o'] = ta.stream_LINEARREG_SLOPE(df['Open'].values, 14)

    df['MEDPRICE'] = ta.MEDPRICE(df['High'], df['Low'])

    return df


def list_of_features():
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'RSI', 'SMA_Fast', 'SMA_Slow', 'BOL_Upp', 'BOL_Mid',
                        'BOL_Low', 'ClosingPctChange',
                        'MOM', 'PlUS_DI_Fast', 'PlUS_DI_Slow', 'LINEARREG_7', 'LINEARREG_14',
                        'MOM', 'PlUS_DI_Fast', 'PlUS_DI_Slow', 'LINEARREG_7', 'LINEARREG_14',
                        'AD', 'LINEARREG_SLOPE_14', 'LINEARREG_SLOPE_7', 'LINEARREG_ANGLE_7', 'LINEARREG_ANGLE_14',
                        'LINEARREG_INTERCEPT_7', 'LINEARREG_INTERCEPT_14',
                        'MEDPRICE', 'MA_5_o', 'MA_20_o', 'BOL_Upp_o', 'BOL_Mid_o', 'BOL_Low_o', 'RSI_o', 'SMA_Fast_o',
                        'SMA_Slow_o', 'MOM_o']

    ''' for open option on linear regression indicator
    , 'LINEARREG_7_o', 'LINEARREG_14_o', 'LINEARREG_ANGLE_7',
                        'LINEARREG_ANGLE_14', 'LINEARREG_INTERCEPT_7_o', 'LINEARREG_INTERCEPT_14_o',
                        'LINEARREG_SLOPE_7_o', 'LINEARREG_SLOPE_14_o'
    '''
    return features
