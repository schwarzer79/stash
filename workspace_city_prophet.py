############################################################
## Import and Setting
############################################################
import logging
logging.getLogger('Prophet').setLevel(logging.ERROR)

# ignore the pystan DeprecationWarning
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )

import os
import sys
import copy
from glob import glob 

import numpy as np

np.random.seed(42)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# folium for interactive mapping of the counters location
import folium
from folium.plugins import MarkerCluster

# some metrics and stats
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import skew

# some utilities from the calendar package
from calendar import day_abbr, month_abbr, mdays

# we use the convenient holiday package from Maurizio Montel to build a DataFrame of national and regional (Auckland region) holidays
import holidays

# fbprophet itself, we use here the version 0.3, release on the 3rd of June 2018
import prophet
prophet.__version__ # 1.1.1

# import some utility functions for data munging and plotting
import utils

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

############################################################
## Import Data
############################################################

train_df = pd.read_csv('dataset/data_city/task1_tr_final_data.csv')
test_df = pd.read_csv('dataset/data_city/task1_ts_final_data.csv')
sample_df = pd.read_csv('dataset/data_city/sample_city.csv')

train_df_origin = pd.read_csv('dataset/data_city/data_tr_city.csv')
test_df_origin = pd.read_csv('dataset/data_city/data_ts_city.csv')

train_df_origin.columns = ['ds','y']
test_df_origin.columns = ['ds','y']

train_df = train_df.drop('Unnamed: 0',axis=1)
test_df = test_df.drop('Unnamed: 0',axis=1)

train_df = pd.concat([train_df['ds'], train_df_origin['y'], train_df.iloc[:,2:]], axis=1)
test_df = pd.concat([test_df['ds'], test_df_origin['y'], test_df.iloc[:,2:]], axis=1)

# 이상값 제거

train_df['ds'] = pd.to_datetime(train_df['ds'])
test_df['ds'] = pd.to_datetime(test_df['ds'])

train_df.info() # 8개 결측값

train_df.loc[train_df['y'] == 0,'y'] = np.NaN
train_df.loc[train_df['y'] > 10000, 'y'] = np.NaN
train_df.loc[train_df['y'] < 0,'y'] = np.NaN

train_df.loc[train_df['y'] > 2000, 'y'] = np.NaN

## Anomaly detection with prophet
# seaborn을 사용하여 데이터 시각화 
sns.set(rc={'figure.figsize':(12,8)}) 
sns.lineplot(x=train_df['ds'], y=train_df['y']) 
plt.legend (['Amount'])
plt.show()

"""
# Add seasonality
model = prophet.Prophet(changepoint_prior_scale = 0.5, seasonality_prior_scale = 0.1, interval_width=0.99, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality = True)
# Fit the model on the training dataset
model.fit(train_df)

prediction = model.predict(train_df)
prediction.plot()
plt.show()

model.plot_components(prediction)
plt.show()

# 실제 값과 예측 값 병합 
performance = pd.merge(train_df, prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
# 이상 징후 개수 확인 
performance['anomaly'].value_counts()

# 이상 현상 살펴보기 
anomalies = performance[performance['anomaly']==1].sort_values(by='ds') 
anomalies

# 이상값 대체
for iter in anomalies.index :
    train_df.loc[train_df.index == iter,'y'] = np.NaN
"""

############################################################
## 결측치 채우기
############################################################


# help function
import copy

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
    df_copy = df_copy.dropna().reset_index()
    return df_copy


# ffill()을 이용한 채우기

train_df_ffill = missing_value_func(train_df, 'ffill')
train_df_ffill = train_df_ffill.drop('index', axis=1)

# bfill()

train_df_bfill = missing_value_func(train_df, 'bfill')
train_df_bfill = train_df_bfill.drop('index', axis=1)

# Simple Moving Average

train_df_SMA = missing_value_func(train_df, 'SMA')
train_df_SMA = train_df_SMA.drop('index', axis=1)

# Exponential Weighted Moving Average

train_df_EWMA = missing_value_func(train_df, 'EWMA')
train_df_EWMA = train_df_EWMA.drop('index', axis=1)

# Interpolation - linear

train_df_Interpolation_linear = missing_value_func(train_df, 'linear')
train_df_Interpolation_linear = train_df_Interpolation_linear.drop('index', axis=1)

# Interpolation - Spline

train_df_Interpolation_spline = missing_value_func(train_df,'spline')
train_df_Interpolation_spline = train_df_Interpolation_spline.drop('index', axis=1)

# Interpolation - time

train_df_Interpolation_time = missing_value_func(train_df, 'time')

train_df_Interpolation_time = train_df_Interpolation_time.drop('index', axis=1)

############################################################
## Cross validation
############################################################

df = pd.concat([train_df_Interpolation_time, test_df], axis=0).reset_index()
df['ds'] = pd.to_datetime(df['ds'])

df = df.drop('index', axis=1)
df = df.drop(['week_weekday', 'kovo_No'], axis=1)

df[df['ds'].dt.year == 2020]

from datetime import datetime
date_after = datetime.strptime('2020-12-31 23:00:00','%Y-%m-%d %H:%M:%S')
date_before = datetime.strptime('2017-01-01 01:00:00','%Y-%m-%d %H:%M:%S')

date_diff = date_after - date_before
date_diff.days

import itertools

param_grid = {  
    'changepoint_range' : [0.8,0.9],
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 3.0, 5.0, 7.0, 10.0],
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mae = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, **params)
    m.add_regressor('population', mode = 'multiplicative')
    m.add_regressor('temp',mode = 'multiplicative')
    m.add_regressor('rain',mode = 'multiplicative')
    m.add_regressor('hum',mode = 'multiplicative')
    m.add_regressor('week_weekday',mode = 'multiplicative')
    m.add_regressor('kovo_Away',mode = 'multiplicative')
    m.add_regressor('kovo_Home',mode = 'multiplicative')
    m.fit(train_df_Interpolation_time)
    df_cv = cross_validation(m, initial='1008 days', period='168 days', horizon='336 days', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    mae.append(df_p['mae'].values[0])

tuning_results = pd.DataFrame(all_params)
tuning_results['mae'] = mae
print(tuning_results)

# Python
best_params = all_params[np.argmin(mae)] # {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0}
print(best_params)

############################################################
## forecast
############################################################

m = prophet.Prophet(changepoint_range=0.9, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1, yearly_seasonality = True, weekly_seasonality=True, daily_seasonality=True)

m.add_regressor('population', mode='multiplicative')
m.add_regressor('temp', mode='multiplicative')
#m.add_regressor('rain', mode='multiplicative')
#m.add_regressor('hum', mode='multiplicative')
m.add_regressor('week_weekend', mode='multiplicative')
#m.add_regressor('kovo_Away', mode='multiplicative')
#m.add_regressor('kovo_Home', mode='multiplicative')

m.fit(train_df_Interpolation_spline)

future = m.make_future_dataframe(periods=336, freq='1H')
future['population'] = df.population
future['temp'] = df.temp
#future['rain'] = df.rain
#future['hum'] = df.hum
future['week_weekend'] = df.week_weekend
#future['kovo_Away'] = df.kovo_Away
#future['kovo_Home'] = df.kovo_Home

forecast = m.predict(future)

pred_y = forecast.yhat.values[-336:]
pred_y_lower = forecast.yhat_lower.values[-336:]
pred_y_upper = forecast.yhat_upper.values[-336:]

y_true = test_df[:336]

from sklearn.metrics import mean_absolute_error

score = mean_absolute_error(y_true['y'], pred_y)
print(score)

"""
# multiplicative

전체 = 97.67967440712407
population 제외 = 98.04583861272734
temp 제외 = 98.4852561978441
rain 제외 = 97.64928414698439
hum 제외 = 97.62721637120521
week_weekend 제외 = 97.85516204993445
kovo_away 제외 = 97.66085026663487

# additive

전체 = 98.38889290775114

"""

###########################
## stochastic 
###########################

copy = train_df
copy = copy.append(test_df.iloc[1,:]).reset_index()
copy.drop('index', axis=1, inplace=True)

def predict_func_stochastic(train_df, test_df) :

    prediction_df = pd.DataFrame([])

    for iter in range(len(test_df)) :
        m = prophet.Prophet(changepoint_range=0.9, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1, yearly_seasonality = True
                            , weekly_seasonality=True, daily_seasonality=True)
        if iter != 0 :
            train_df = train_df.append(test_df.iloc[iter-1,:])
        m.fit(train_df)
        future = m.make_future_dataframe(periods=336, freq='1H')
        forecast = m.predict(future)
        pred_y = forecast.yhat.values[-336:]
        prediction_df = prediction_df.append(pd.Series(pred_y), ignore_index=True)
        
    return prediction_df

pred_df = predict_func_stochastic(train_df, test_df)