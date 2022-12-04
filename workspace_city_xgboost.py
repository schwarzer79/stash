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

#####################
## data import
#####################
train_df = pd.read_csv('dataset/data_city/task1_city_tr_data.csv',index_col=0)
test_df = pd.read_csv('dataset/data_city/task1_ts_final_data.csv',index_col=0)
sample_df = pd.read_csv('dataset/data_city/sample_city.csv',index_col=0)
test_df_xgb = pd.read_csv('dataset/data_city/city_ts_final.csv',index_col=0)

train_df.drop(['kovo_Away', 'kovo_Home', 'kovo_No', 'week_weekend'], axis=1, inplace=True)
test_df.drop(['kovo_Away', 'kovo_Home', 'kovo_No', 'week_weekend'], axis=1, inplace=True)
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
    train_df.drop(['weekday'], inplace=True, axis=1)

    # dtype 변경
    cat_cols = ['time_of_day','season','time']
    #cat_cols = ['hour','weekday','time_of_day']
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')

    # get_dummies
    train_df = pd.get_dummies(train_df, drop_first=True)

    return train_df

train_df = feature_preprocessing(train_df)
test_df = feature_preprocessing(test_df)
test_df_xgb = feature_preprocessing(test_df_xgb)

train_df.columns
test_df.columns
test_df_xgb.columns

####################
## outlier + missing value
####################

# 결측값 개수 확인
train_df.isna().sum() # 'y' : 8
test_df.isna().sum()

# 값 분포 확인
train_df.describe() # 극단적으로 절댓값이 큰 값들이 존재

# outlier를 missing value로 대체
train_df.loc[(abs(train_df['y']) > 2500) | (train_df['y'] == 0), 'y'] = np.NaN # 이상치를 missing value로 전환 -> 153개의 missing value 생성

# index를 ds로 변경
train_df.set_index('ds', inplace=True)
train_df.head()

test_df.set_index('ds', inplace=True)
test_df_xgb.set_index('ds',inplace=True)

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


########################################################################################
## Cross Validation
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df.copy()
test_df_xgb_final = test_df_xgb.copy()

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'n_estimators': [10, 31, 52, 73, 94, 115, 136, 157, 178, 200]}

"""
xgbtuned = xgb.XGBRegressor()
tscv = TimeSeriesSplit(n_splits=5)
xgbtunedreg = RandomizedSearchCV(xgbtuned, param_distributions=param_grid , 
                                   scoring='neg_mean_squared_error', n_iter=20, n_jobs=-1, 
                                   cv=tscv, verbose=2, random_state=42)

X_train = train_df_final.drop(['y', 'day'], axis=1)
y_train = train_df_final.y

X_test = test_df_xgb_final.drop(['y', 'day'], axis=1)
y_test = test_df_xgb_final.y

X_test.columns
X_train.columns

xgbtunedreg.fit(X_train, y_train)

best_score = xgbtunedreg.best_score_
best_params = xgbtunedreg.best_params_
print("Best score: {}".format(best_score))
print("Best params: {}".format(best_params))

# Best params: {'subsample': 0.9, 'n_estimators': 31, 'min_child_weight': 3.0, 'max_depth': 4, 'learning_rate': 0.2, 'gamma': 1.0, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9}
preds_boost_tuned = xgbtunedreg.predict(X_test)
"""
# _ = error_metrics(preds_boost_tuned, y_test, model_name='Tuned XGBoost with Fourier terms', test=True)
"""
Error metrics for model Tuned XGBoost with Fourier terms
RMSE or Root mean squared error: 109.57
Variance score: 0.53
Mean Absolute Error: 86.00
Mean Absolute Percentage Error: 53.15 %
"""

# _ = error_metrics(xgbtunedreg.predict(X_train), y_train, model_name='Tuned XGBoost with Fourier terms', test=False)
"""
Error metrics for model Tuned XGBoost with Fourier terms
RMSE or Root mean squared error: 119.84
Variance score: 0.48
Mean Absolute Error: 89.67
Mean Absolute Percentage Error: 71.52 %
"""

def xgboost_cross_validation(train_df, test_df, params) :

    xgbtuned = xgb.XGBRegressor()
    tscv = TimeSeriesSplit(n_splits=7)
    xgbtunedreg = RandomizedSearchCV(xgbtuned, param_distributions=params , 
                                   scoring='neg_mean_squared_error', n_iter=20, n_jobs=-1, 
                                   cv=tscv, verbose=2, random_state=42)

    X_train = train_df.drop('y', axis=1)
    y_train = train_df.y

    X_test = test_df.drop('y', axis=1)
    y_test = test_df.y

    xgbtunedreg.fit(X_train, y_train)
    
    best_score = xgbtunedreg.best_score_
    best_params = xgbtunedreg.best_params_
    print("Best score: {}".format(best_score))
    print("Best params: {}".format(best_params))

    preds_boost_tuned = xgbtunedreg.predict(X_test)

    error_metrics(preds_boost_tuned, y_test, model_name='Tuned XGBoost with Fourier terms', test=True)
    error_metrics(xgbtunedreg.predict(X_train), y_train, model_name='Tuned XGBoost with Fourier terms', test=False)

    return preds_boost_tuned, X_test

#########################################################################################
## Naive forecast 
#########################################################################################
train_df_final = train_df_Interpolation_linear.copy()
test_df_final = test_df_xgb.copy()

train_df_final.info()

train_df_naive = train_df_final.drop(['day','rain','time_evening','time_morning','time_night'], axis=1)
train_df_naive = train_df_naive.dropna()
test_df_naive = test_df_final.drop(['day','rain','time_evening','time_morning','time_night'], axis=1)

train_df_naive.info()

pred_y, X_test = xgboost_cross_validation(train_df_naive,test_df_naive,param_grid)

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

result.to_csv('result/xgb_naive_origin_rain.csv')

#########################################################################################
## add 24 Time lag -> 108
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

## lag variable 생성
for i in range(48):
    train_df_final['lag'+str(i+1)] = train_df_final['y'].shift(i+1)

for i in range(48):
    test_df_final['lag'+str(i+1)] = test_df_final['y'].shift(i+1)

train_df_lag = train_df_final.drop(['day'], axis=1)
train_df_lag = train_df_lag.dropna()
test_df_lag = test_df_final.drop(['day'], axis=1)

pred_y , X_test = xgboost_cross_validation(train_df_lag, test_df_lag, param_grid)

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
sample_df.drop('datetime', axis=1, inplace=True)
result.columns = sample_df.columns

result.to_csv('xgb_1to24_lag.csv')
sample_df

result.shape

#########################################################################################
## add only 24 lag
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

## lag variable 생성
train_df_final['lag24'] = train_df_final['y'].shift(24)
test_df_final['lag24'] = test_df_final['y'].shift(24)

train_df_final.columns

# na 제거
train_df_final.dropna(inplace=True)

# 필요없는 열 제거
train_df_lag = train_df_final.drop(['day'], axis=1)
test_df_lag = test_df_final.drop(['day'], axis=1)

pred_y, X_test = xgboost_cross_validation(train_df_lag, test_df_lag, param_grid)

# 결과 set 만들기
result = []
for iter in range(len(X_test_24)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test_24.index[:8425])
sample_df.drop('datetime', axis=1, inplace=True)
result.columns = sample_df.columns

result.to_csv('result/xgb_24_lag.csv')


#########################################################################################
## fouier
#########################################################################################

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

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

## fourier
test_df_final.index = pd.to_datetime(test_df_final.index)
add_fourier_terms(train_df_final, year_k= 5, week_k=5 , day_k=5)
add_fourier_terms(test_df_final, year_k= 5, week_k=5 , day_k=5)

test_df_final.columns

# 필요없는 열 제거
train_df_fourier = train_df_final.drop(['day'], axis=1)
test_df_fourier = test_df_final.drop(['day'], axis=1)

pred_y, X_test = xgboost_cross_validation(train_df_fourier, test_df_fourier, param_grid)

# 결과 set 만들기
result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
sample_df.drop('datetime', axis=1, inplace=True)
result.columns = sample_df.columns

result.to_csv('result/xgb_fourier.csv')

#########################################################################################
## lag 1 ~ 24 + other feature
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

## lag variable 생성
for i in range(24):
    train_df_final['lag'+str(i+1)] = train_df_final['y'].shift(i+1)

for i in range(24):
    test_df_final['lag'+str(i+1)] = test_df_final['y'].shift(i+1)

train_df_lag = train_df_final.drop(['day','rain','time_evening','time_morning','time_night'], axis=1)
train_df_lag = train_df_lag.dropna()
test_df_lag = test_df_final.drop(['day','rain','time_evening','time_morning','time_night'], axis=1)

def add_feature(df) :

    df['rolling_6_min'] = df['y'].shift(1).rolling(6).min()
    df['rolling_12_min'] = df['y'].shift(1).rolling(12).min()
    df['rolling_24_min'] = df['y'].shift(1).rolling(24).min()

    df['rolling_6_max'] = df['y'].shift(1).rolling(6).max()
    df['rolling_12_max'] = df['y'].shift(1).rolling(12).max()
    df['rolling_24_max'] = df['y'].shift(1).rolling(24).max()

    df['rolling_6_mean'] = df['y'].shift(1).rolling(6).mean()
    df['rolling_12_mean'] = df['y'].shift(1).rolling(12).mean()
    df['rolling_24_mean'] = df['y'].shift(1).rolling(24).mean()

    df['rolling_6_std'] = df['y'].shift(1).rolling(6).std()
    df['rolling_12_std'] = df['y'].shift(1).rolling(12).std()
    df['rolling_24_std'] = df['y'].shift(1).rolling(24).std()

    return df

train_df_mul = add_feature(train_df_lag)
test_df_mul = add_feature(test_df_lag)
train_df_mul.dropna(inplace=True)

train_df_mul.isna()

pred_y, X_test = xgboost_cross_validation(train_df_mul, test_df_mul, param_grid)

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

result.to_csv('result/xgb_lag_mul.csv')

#########################################################################################
## lag 1 ~ 24 + fourier + rolling mean, std + hour, month, year 삭제
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

## lag variable 생성
for i in range(24):
    train_df_final['lag'+str(i+1)] = train_df_final['y'].shift(i+1)

for i in range(24):
    test_df_final['lag'+str(i+1)] = test_df_final['y'].shift(i+1)

train_df_lag = train_df_final.drop(['day','rain','time_evening','time_morning','time_night', 'dayofyear'], axis=1)
train_df_lag = train_df_lag.dropna()
test_df_lag = test_df_final.drop(['day','rain','time_evening','time_morning','time_night', 'dayofyear'], axis=1)

def add_feature(df) :

    df['rolling_6_min'] = df['y'].shift(1).rolling(6).min()
    df['rolling_12_min'] = df['y'].shift(1).rolling(12).min()
    df['rolling_24_min'] = df['y'].shift(1).rolling(24).min()

    df['rolling_6_max'] = df['y'].shift(1).rolling(6).max()
    df['rolling_12_max'] = df['y'].shift(1).rolling(12).max()
    df['rolling_24_max'] = df['y'].shift(1).rolling(24).max()

    #df['rolling_6_mean'] = df['y'].shift(1).rolling(6).mean()
    #df['rolling_12_mean'] = df['y'].shift(1).rolling(12).mean()
    #df['rolling_24_mean'] = df['y'].shift(1).rolling(24).mean()

    #df['rolling_6_std'] = df['y'].shift(1).rolling(6).std()
    #df['rolling_12_std'] = df['y'].shift(1).rolling(12).std()
    #df['rolling_24_std'] = df['y'].shift(1).rolling(24).std()

    df['rolling_6_med'] = df['y'].shift(1).rolling(6).median()
    df['rolling_12_med'] = df['y'].shift(1).rolling(12).median()
    df['rolling_24_med'] = df['y'].shift(1).rolling(24).median()

    return df

test_df_lag.index = pd.to_datetime(test_df_lag.index)
train_df_lag = add_fourier_terms(train_df_lag, year_k= 5, week_k=5, day_k=5)
test_df_lag = add_fourier_terms(test_df_lag, year_k= 5, week_k=5, day_k=5)

train_df_mul = add_feature(train_df_lag)
test_df_mul = add_feature(test_df_lag)
train_df_mul.dropna(inplace=True)

train_df_mul.columns

pred_y, X_test = xgboost_cross_validation(train_df_mul, test_df_mul, param_grid)

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

result.to_csv('result/xgb_lag_mul_fourier.csv')


#########################################################################################
## lag 168 + fourier + rolling mean, std 
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

## lag variable 생성
for i in range(24):
    train_df_final['lag'+str(i+1)] = train_df_final['y'].shift(i+1)

for i in range(24):
    test_df_final['lag'+str(i+1)] = test_df_final['y'].shift(i+1)

#train_df_final['lag'+str(24*7)] = train_df_final['y'].shift(24*7)
#test_df_final['lag'+str(24*7)] = test_df_final['y'].shift(24*7)

train_df_lag = train_df_final.drop(['day','rain','time_evening','time_morning','time_night', 'dayofyear'], axis=1)
train_df_lag = train_df_lag.dropna()
test_df_lag = test_df_final.drop(['day','rain','time_evening','time_morning','time_night', 'dayofyear'], axis=1)

def add_feature(df) :

    #df['rolling_6_min'] = df['y'].shift(1).rolling(6).min()
    #df['rolling_12_min'] = df['y'].shift(1).rolling(12).min()
    #df['rolling_24_min'] = df['y'].shift(1).rolling(24).min()

    #df['rolling_6_max'] = df['y'].shift(1).rolling(6).max()
    #df['rolling_12_max'] = df['y'].shift(1).rolling(12).max()
    #df['rolling_24_max'] = df['y'].shift(1).rolling(24).max()

    df['rolling_6_mean'] = df['y'].shift(1).rolling(6).mean()
    df['rolling_12_mean'] = df['y'].shift(1).rolling(12).mean()
    df['rolling_24_mean'] = df['y'].shift(1).rolling(24).mean()

    df['rolling_6_std'] = df['y'].shift(1).rolling(6).std()
    df['rolling_12_std'] = df['y'].shift(1).rolling(12).std()
    df['rolling_24_std'] = df['y'].shift(1).rolling(24).std()

    #df['rolling_6_med'] = df['y'].shift(1).rolling(6).median()
    #df['rolling_12_med'] = df['y'].shift(1).rolling(12).median()
    #df['rolling_24_med'] = df['y'].shift(1).rolling(24).median()

    return df

test_df_lag.index = pd.to_datetime(test_df_lag.index)
train_df_lag = add_fourier_terms(train_df_lag, year_k= 5, week_k=5, day_k=5)
test_df_lag = add_fourier_terms(test_df_lag, year_k= 5, week_k=5, day_k=5)

train_df_mul = add_feature(train_df_lag)
test_df_mul = add_feature(test_df_lag)
train_df_mul.dropna(inplace=True)

train_df_mul.columns

pred_y, X_test = xgboost_cross_validation(train_df_mul, test_df_mul, param_grid)

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

result.to_csv('result/xgb_lag_mul_fourier.csv')

########################################################
## prophet + LightGBM
########################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

train_df_lag = train_df_final.drop(['day','rain','time_evening','time_morning','time_night', 'dayofyear'], axis=1)
test_df_lag = test_df_final.drop(['day','rain','time_evening','time_morning','time_night', 'dayofyear'], axis=1)

train_df_lag['ds'] = train_df_lag.index
test_df_lag['ds'] = test_df_lag.index


import prophet

m = prophet.Prophet(
                growth='linear',
                seasonality_mode='multiplicative',
                interval_width=0.95,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,changepoint_range=0.9, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1
            )

m.fit(train_df_lag)
#extract features from data using prophet to predict train set
predictions_train = m.predict(train_df_lag.drop('y', axis=1))
#extract features from data using prophet to predict test set
predictions_test = m.predict(test_df_lag.drop('y', axis=1))
#merge train and test predictions
predictions = pd.concat([predictions_train, predictions_test], axis=0)

df = pd.concat([train_df_lag,test_df_lag],axis=0)
df['aa'] = range(len(df))
df.set_index('aa',inplace=True)

df = pd.merge(df, predictions,  how='inner')

for i in range(24):
   df['lag'+str(i+1)] = df['y'].shift(i+1)

df.dropna(inplace=True)
df.drop('ds',axis=1,inplace=True)

horizon = len(test_df_xgb)
X = df.drop('y',axis=1)
y = df.y
X_train, X_test = X.iloc[:-horizon,:], X.iloc[-horizon:,:]
y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

import optuna  # pip install optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from optuna.integration import LightGBMPruningCallback

def objective(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgbm.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],  # Add a pruning callback
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)

import lightgbm
#define LightGBM model, train it and make predictions
model = lightgbm.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = np.round(mean_absolute_error(y_test, predictions), 3) 