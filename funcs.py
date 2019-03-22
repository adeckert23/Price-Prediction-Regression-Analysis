import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor

def linear_regress(X_train, y_train, X_test):
    model = LinearRegression()
    result = model.fit(X_train,y_train)
    y_predicted = model.predict(X_test)
    return y_predicted

def run_ridge(X_train,y_train):
    model = Ridge(alpha=.1)
    result = model.fit(X_train,y_train)
    y_predicted = model.predict(X_train)
    return y_predicted


def run_lasso(X_train, y_train, X_test):
    model = Lasso(alpha=1)
    result = model.fit(X_train,y_train)
    y_predicted = model.predict(X_test)
    return y_predicted

`need dataframe with few features`
def kneighbs(X_train, y_train, X_test):
    model = KNeighborsRegressor(n_neighbors = 10, weights= 'distance')
    result = model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    return y_predicted

def to_csv(y_pred, df):
    df['SalePrice'] = y_pred
    df[['SalesID','SalePrice']].to_csv('data/output.csv', sep=',', index = False)
