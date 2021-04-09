from __future__ import print_function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import data_from_Tech_Analyze as dTA


def make_prediction(quotes_df, estimator):
    # Make a copy of the dataframe so we don't modify the original
    df = quotes_df.copy()
    df = dTA.technical_indicators(df)
    featureList = dTA.list_of_features()
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
    features_to_fit = featureList

    # Create our target and labels
    X = df[features_to_fit]
    y = df['NextClose']

    # Create training and testing data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True,
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


def predictors(quotes):
    avgprice = 0
    quotes_df = quotes.copy()
    # Predict the last day's closing price using linear regression
    print('Unscaled Linear Regression:')
    linreg = LinearRegression()
    ln = make_prediction(quotes_df, linreg)
    avgprice += ln
    print('Predicted Closing Price: %.2f\n' % ln)

    # Predict the last day's closing price using Linear regression with scaled features
    print('Scaled Linear Regression:')
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    p = make_prediction(quotes_df, pipe)
    avgprice += p
    print('Predicted Closing Price: %.2f\n' % p)

    # Predict the last day's closing price using ridge regression
    print('Unscaled Ridge Regression:')
    ridge = Ridge(normalize=True)
    r = make_prediction(quotes_df, ridge)
    avgprice += r
    print('Predicted Closing Price: %.2f\n' % r)

    # Predict the last day's closing price using ridge regression and scaled features
    print('Scaled Linear Regression:')
    ridge_pipe = make_pipeline(StandardScaler(), Ridge())
    rr = make_prediction(quotes_df, ridge_pipe)
    avgprice += rr
    print('Predicted Closing Price: %.2f\n' % rr)

    # Predict the last day's closing price using decision tree regression
    print('Unscaled Decision Tree Regressor:')
    tree = DecisionTreeRegressor()
    t = make_prediction(quotes_df, tree)
    avgprice += t
    print('Predicted Closing Price: %.2f\n' % t)

    avgprice /= 5
    print("Average Predicted Price: %.2f\n" % avgprice)
    return