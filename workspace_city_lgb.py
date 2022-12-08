#####################
## import libraries
#####################
# importing all the required libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import scipy.stats
import matplotlib.dates as mdates
import copy
plt.style.use('bmh')
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import random
random.seed(42)

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb

##################
## help function
###################
dict_error = dict()

def error_metrics(y_pred, y_truth, model_name = None, test = True):
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred
    else:
        y_pred = y_pred.to_numpy()
        
    if isinstance(y_truth, np.ndarray):
        y_truth = y_truth
    else:
        y_truth = y_truth.to_numpy()
        
    print('\nError metrics for model {}'.format(model_name))
    
    RMSE = np.sqrt(mean_squared_error(y_truth, y_pred))
    print("RMSE or Root mean squared error: %.2f" % RMSE)
    
    # Explained variance score: 1 is perfect prediction

    R2 = r2_score(y_truth, y_pred)
    print('Variance score: %.2f' % R2 )

    MAE = mean_absolute_error(y_truth, y_pred)
    print('Mean Absolute Error: %.2f' % MAE)

    MAPE = (np.mean(np.abs((y_truth - y_pred) / y_truth)) * 100)
    print('Mean Absolute Percentage Error: %.2f %%' % MAPE)
    
    # Appending the error values along with the model_name to the dict
    if test:
        train_test = 'test'
    else:
        train_test = 'train'
    
    #df = pd.DataFrame({'model': model_name, 'RMSE':RMSE, 'R2':R2, 'MAE':MAE, 'MAPE':MAPE}, index=[0])
    name_error = ['model', 'train_test', 'RMSE', 'R2', 'MAE', 'MAPE']
    value_error = [model_name, train_test, RMSE, R2, MAE, MAPE]
    list_error = list(zip(name_error, value_error))
    
    for error in list_error:
        if error[0] in dict_error:
            # append the new number to the existing array at this slot
            dict_error[error[0]].append(error[1])
        else:
            # create a new array in this slot
            dict_error[error[0]] = [error[1]]
    #return(dict_error)

# hour 를 night, morning, afternoon, evening으로 분류
hour_dict = {'morning': list(np.arange(6,12)),'afternoon': list(np.arange(12,18)), 'evening': list(np.arange(18,24)),
            'night': [0, 1, 2, 3, 4, 5]}

time_dict = {'morning': list(np.arange(6,9)),'afternoon': list(np.arange(12,15)), 'evening': list(np.arange(18,22)),
            'night': [0, 1, 2, 3, 4, 5]}


# 적용
def time_of_day(x):
    if x in hour_dict['morning']:
        return 'morning'
    elif x in hour_dict['afternoon']:
        return 'afternoon'
    elif x in hour_dict['evening']:
        return 'evening'
    else:
        return 'night'

def time(x):
    if x in time_dict['morning']:
        return 'morning'
    elif x in time_dict['afternoon']:
        return 'afternoon'
    elif x in time_dict['evening']:
        return 'evening'
    else:
        return 'night'

def add_fourier_terms(df, year_k, week_k, day_k):
    
    for k in range(1, year_k+1):
        # year has a period of 365.25 including the leap year
        df['year_sin'+str(k)] = np.sin(2 *k* np.pi * df.index.dayofyear/365.25) 
        df['year_cos'+str(k)] = np.cos(2 *k* np.pi * df.index.dayofyear/365.25)

    for k in range(1, week_k+1):
        
         # week has a period of 7
        df['week_sin'+str(k)] = np.sin(2 *k* np.pi * df.index.dayofweek/7)
        df['week_cos'+str(k)] = np.cos(2 *k* np.pi * df.index.dayofweek/7)


    for k in range(1, day_k+1):
        
        # day has period of 24
        df['hour_sin'+str(k)] = np.sin(2 *k* np.pi * df.index.hour/24)
        df['hour_cos'+str(k)] = np.cos(2 *k* np.pi * df.index.hour/24)

    return df


#####################
## data import
#####################
train_df = pd.read_csv('dataset/data_city/task1_city_tr_data.csv',index_col=0)
#test_df = pd.read_csv('dataset/data_city/task1_ts_final_data.csv',index_col=0)
sample_df = pd.read_csv('dataset/data_city/sample_city.csv',index_col=0)
test_df_xgb = pd.read_csv('dataset/data_city/city_ts_final.csv',index_col=0)

train_df.drop('index', axis=1, inplace=True)

#train_df.drop(['kovo_Away', 'kovo_Home', 'kovo_No', 'week_weekend'], axis=1, inplace=True)
#test_df.drop(['kovo_Away', 'kovo_Home', 'kovo_No', 'week_weekend'], axis=1, inplace=True)
test_df_xgb.drop(['year','month','day','hour'], axis=1, inplace=True)

def season_calc(month):
    if month in [5,6,7,8]:
        return "summer"
    else:
        return "winter"

def is_season(ds) :
    date = pd.to_datetime(ds)
    return (date.month > 10 or date.month < 4)

def feature_preprocessing(train_df) :
    # 'ds' 데이터형 변경
    train_df.ds = pd.to_datetime(train_df['ds'])

    # ds 기반 새로운 feature 생성
    train_df['month'] = train_df.ds.dt.month
    train_df['year'] = train_df.ds.dt.year
    train_df['day'] = train_df.ds.dt.day
    train_df['hour'] = train_df.ds.dt.hour
    weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
    train_df['weekday'] = train_df.ds.dt.weekday.map(weekdays)

    train_df['dayofyear'] = train_df.ds.dt.dayofyear

    # 계절에 관한 feature 생성
    for iter in range(len(train_df)) :
        train_df.loc[iter,'season'] = season_calc(train_df.loc[iter, 'month'])

    # hour 변환
    train_df['time_of_day'] = train_df['hour'].apply(time_of_day)
    train_df['time'] = train_df['hour'].apply(time)

    # hour 삭제
    # train_df.drop(['hour'], inplace=True, axis=1)
    # train_df.drop(['weekday'], inplace=True, axis=1)

    # dtype 변경
    cat_cols = ['time_of_day','season','time','weekday','holiday','warning']
    #cat_cols = ['hour','weekday','time_of_day']
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')

    # get_dummies
    train_df = pd.get_dummies(train_df, drop_first=True)

    return train_df

train_df.columns
# test_df.columns
test_df_xgb.columns

####################
## outlier + missing value
####################

# 결측값 개수 확인
train_df.isna().sum() # 'y' : 8
test_df_xgb.isna().sum()

# 값 분포 확인
#train_df.describe() # 극단적으로 절댓값이 큰 값들이 존재
#train_df.describe(include='o')

# outlier를 missing value로 대체
train_df.loc[(abs(train_df['y']) > 2000) | (train_df['y'] == 0), 'y'] = np.NaN # 이상치를 missing value로 전환 -> 153개의 missing value 생성
train_df['y'].isna().sum()

# index를 ds로 변경
#train_df.set_index('ds', inplace=True)
train_df.head()
#test_df_xgb.set_index('ds',inplace=True)

#train_df['y'].plot()
#plt.show()

# help function
def missing_value_func(df, method, window = 5, halflife = 4) :
    df_copy = copy.deepcopy(df)
    if method == 'ffill' :
        df_copy['y'] = df_copy['y'].fillna(method = method)
    elif method == 'bfill' :
        df_copy['y'] = df_copy['y'].fillna(method = method)
    elif method == 'SMA' :
        df_copy['y'] = df_copy['y'].rolling(window=window, min_periods=1).mean()
    elif method == 'WMA' :
        df_copy['y'] = df_copy['y'].ewm(halflife=halflife).mean()
    elif method == 'linear' :
        df_copy['y'] = df_copy['y'].interpolate(option=method)
    elif method == 'spline' :
        df_copy['y'] = df_copy['y'].interpolate(option=method)
    else :
        df_copy['y'] = df_copy['y'].interpolate(option=method)
    df_copy = df_copy.dropna()
    return df_copy

# ffill()을 이용한 채우기

train_df_ffill = missing_value_func(train_df, 'ffill')

# bfill()

train_df_bfill = missing_value_func(train_df, 'bfill')

# Simple Moving Average

train_df_SMA = missing_value_func(train_df, 'SMA')

# Exponential Weighted Moving Average

train_df_EWMA = missing_value_func(train_df, 'EWMA')

# Interpolation - linear

train_df_Interpolation_linear = missing_value_func(train_df, 'linear')

# Interpolation - Spline

train_df_Interpolation_spline = missing_value_func(train_df,'spline')

# Interpolation - time

train_df_Interpolation_time = missing_value_func(train_df, 'time')

train_df_Interpolation_time = feature_preprocessing(train_df_Interpolation_time)
# test_df = feature_preprocessing(test_df)
test_df_xgb = feature_preprocessing(test_df_xgb)

train_df_Interpolation_time.set_index('ds', inplace=True)
test_df_xgb.set_index('ds',inplace=True)

#################################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

X_df = train_df_final.drop('y', axis=1)
y_df = train_df_final.y

X_test = test_df_final.drop('y', axis=1)
y_test = test_df_final.y

X_train = X_df.loc[X_df.index < '2020-01-01 00:00:00',:]
y_train = y_df.loc[y_df.index < '2020-01-01 00:00:00']

X_val = X_df.loc[X_df.index >= '2020-01-01 00:00:00',:]
y_val = y_df.loc[y_df.index >= '2020-01-01 00:00:00']

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

params = {
    'n_estimators': 2000,
    'max_depth': 4,
    'num_leaves': 2**4,
    'learning_rate': 0.1,
    'boosting_type': 'dart'
}

model = lgb.LGBMRegressor(first_metric_only = True, **params)

model.fit(X_train, y_train,
          eval_metric = 'l1', 
          eval_set = [(X_val, y_val)],
          #early_stopping_rounds = 10,
          verbose = 0)

forecast = model.predict(X_test)

mae = mean_absolute_error(y_test,forecast)

######################################################

model = lgb.LGBMRegressor()
model.get_params()

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=50,
          eval_metric='mae',
          early_stopping_rounds=20)

params = {'objective' : ['regression', 'huber'],
          'boosting' : ['gdbt','dart','rf'],
          'learning_rate' : [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
          'num_leaves' : [31, 50, 100, 150, 300, 600, 1000, 3000],
          'device_type' : ['cuda_exp'],
          'seed' : [42],
          'max_depth' : [3,5,7,9,13,15,20],
          'min_data_in_leaf' : [20, 30, 50],
          'bagging_fraction' : [0.1,0.2,0.3],
          'bagging_freq' : [25,50],
          'feature_fraction_seed' : [42],
          'extra_trees' : True,
          'extra_seed' : 42,
          'early_stopping_round' : 35,
          'metric' : ['mae'],
          }

model = lgb.LGBMRegressor(**params)
lgb.cv(params, X_train, shuffle=False,)