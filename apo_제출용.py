#####################
## import libraries
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import datetime as dt
from pytimekr import pytimekr # 휴일 library

from datetime import datetime, timedelta
import matplotlib.dates as mdates
import copy
plt.style.use('bmh')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import random
import numpy as np
import random
import os

# random seed 42
def my_seed_everywhere(seed: int = 42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os

my_seed = 42
my_seed_everywhere(my_seed)

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

##################
## help function
###################
dict_error = dict()

# 자체 모델 평가를 위한 평가지표 function
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

# time 변수 생성 함수
def time(x):
    if x in time_dict['morning']:
        return 'morning'
    elif x in time_dict['afternoon']:
        return 'afternoon'
    elif x in time_dict['evening']:
        return 'evening'
    else:
        return 'night'

# 푸리에 생성
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

# CV용 함수
def xgboost_cross_validation(train_df, test_df, params, n_iter=100) :

    xgbtuned = xgb.XGBRegressor(tree_method='gpu_hist')
    tscv = TimeSeriesSplit(n_splits=5)
    xgbtunedreg = RandomizedSearchCV(xgbtuned, param_distributions=params , 
                                   scoring='neg_mean_absolute_error', n_iter=n_iter, n_jobs=-1, 
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
    #error_metrics(xgbtunedreg.predict(X_train), y_train, model_name='Tuned XGBoost with Fourier terms', test=False)

    return preds_boost_tuned, X_test

# 추가적인 feature를 위한 함수
def add_feature(df) :

    df['rolling_6_mean'] = df['y'].shift(1).rolling(6).mean()
    df['rolling_12_mean'] = df['y'].shift(1).rolling(12).mean()
    df['rolling_24_mean'] = df['y'].shift(1).rolling(24).mean()

    df['rolling_6_std'] = df['y'].shift(1).rolling(6).std()
    df['rolling_12_std'] = df['y'].shift(1).rolling(12).std()
    df['rolling_24_std'] = df['y'].shift(1).rolling(24).std()
    
    return df

##############################################################################
## 데이터 전처리
##############################################################################
# -*- coding: utf-8 -*-
"""
task1_prepro_sw.py

@author: Sewon
"""
# path 설정 -----------------------------------------------

path = 'dataset/data_apo' # 적산차 값을 넣기 위한 path 설정
c = pd.read_csv(path + '/data_tr_apo.csv')
d = pd.read_csv(path + '/data_ts_apo.csv')

train = c.copy()
test = d.copy()

train.columns = ['ds', 'y'] # 컬럼명 변경
test.columns = ['ds', 'y']

train.ds = pd.to_datetime(train.ds) 
test.ds = pd.to_datetime(test.ds)

# =============================================================================
#  test set 기간 추가 ( ~ 2022.01.01 00:00:00) / xgboost를 이용하기에 예측하고자 하는 전구간에 대해 date 생성
# =============================================================================
s = pd.Series(pd.date_range("2021-12-18 00:00:00", periods=337, freq = '0D1H0min0S'))

df_s = pd.DataFrame({'ds' : s})

test = pd.concat([test, df_s])
test = test.reset_index()
test

# =============================================================================
# week_weekend 변수 추가
# =============================================================================
train.ds.dt.weekday.unique() 
'''
6 : 일, 0 : 월, 1 : 화, 2 : 수, 3 : 목, 4 : 금, 5 : 토
'''

train['week'] = 'weekday' # weekday로 초기화
train.loc[(train.ds.dt.weekday == 5) | (train.ds.dt.weekday == 6), 'week'] = 'weekend'
train = pd.get_dummies(train, 'week')
train = train.drop('week_weekend', axis = 1)

test['week'] = 'weekday' # weekday로 초기화
test.loc[(test.ds.dt.weekday == 5) | (test.ds.dt.weekday == 6), 'week'] = 'weekend'
test = pd.get_dummies(test, 'week')
test = test.drop('week_weekend', axis = 1)

####################################################################################
## 인구
####################################################################################

# 인구수 아포읍

kim2017 = pd.read_csv('population/population_2017.csv', encoding='cp949', thousands=',')
kim2018 = pd.read_csv('population/population_2018.csv', encoding='cp949', thousands=',')
kim2019 = pd.read_csv('population/population_2019.csv', encoding='cp949', thousands=',')
kim2020 = pd.read_csv('population/population_2020.csv', encoding='cp949', thousands=',')
kim2021 = pd.read_csv('population/population_2021.csv', encoding='cp949', thousands=',')
combine = [kim2017, kim2018, kim2019, kim2020, kim2021]

year_2017=pd.DataFrame(kim2017)
year_2018=pd.DataFrame(kim2018)
year_2019=pd.DataFrame(kim2019)
year_2020=pd.DataFrame(kim2020)
year_2021=pd.DataFrame(kim2021)

final_apo2017=year_2017[['2017년01월_총인구수','2017년02월_총인구수','2017년03월_총인구수','2017년04월_총인구수',
             '2017년05월_총인구수','2017년06월_총인구수','2017년07월_총인구수','2017년08월_총인구수',
             '2017년09월_총인구수','2017년10월_총인구수','2017년11월_총인구수','2017년12월_총인구수']]

final_apo2018=year_2018[['2018년01월_총인구수','2018년02월_총인구수','2018년03월_총인구수','2018년04월_총인구수',
             '2018년05월_총인구수','2018년06월_총인구수','2018년07월_총인구수','2018년08월_총인구수',
             '2018년09월_총인구수','2018년10월_총인구수','2018년11월_총인구수','2018년12월_총인구수']]

final_apo2019=year_2019[['2019년01월_총인구수','2019년02월_총인구수','2019년03월_총인구수','2019년04월_총인구수',
             '2019년05월_총인구수','2019년06월_총인구수','2019년07월_총인구수','2019년08월_총인구수',
             '2019년09월_총인구수','2019년10월_총인구수','2019년11월_총인구수','2019년12월_총인구수']]

final_apo2020=year_2020[['2020년01월_총인구수','2020년02월_총인구수','2020년03월_총인구수','2020년04월_총인구수',
             '2020년05월_총인구수','2020년06월_총인구수','2020년07월_총인구수','2020년08월_총인구수',
             '2020년09월_총인구수','2020년10월_총인구수','2020년11월_총인구수','2020년12월_총인구수']]

final_apo2021=year_2021[['2021년01월_총인구수','2021년02월_총인구수','2021년03월_총인구수','2021년04월_총인구수',
             '2021년05월_총인구수','2021년06월_총인구수','2021년07월_총인구수','2021년08월_총인구수',
             '2021년09월_총인구수','2021년10월_총인구수','2021년11월_총인구수','2021년12월_총인구수']]

final_apo2017=final_apo2017.loc[[1]] # 해당 자료에서 각 지역에 해당하는 행을 지정 / 여기서는 1
final_apo2018=final_apo2018.loc[[1]]
final_apo2019=final_apo2019.loc[[1]]
final_apo2020=final_apo2020.loc[[1]]
final_apo2021=final_apo2021.loc[[1]]

new_2017=final_apo2017.rename(index={1:'아포읍 인구수'})
new_2018=final_apo2018.rename(index={1:'아포읍 인구수'})
new_2019=final_apo2019.rename(index={1:'아포읍 인구수'})
new_2020=final_apo2020.rename(index={1:'아포읍 인구수'})
new_2021=final_apo2021.rename(index={1:'아포읍 인구수'})

new_2017=new_2017.T
new_2018=new_2018.T
new_2019=new_2019.T
new_2020=new_2020.T
new_2021=new_2021.T

fin_apo =pd.concat([new_2017,new_2018,new_2019,new_2020,new_2021],axis=0)

fin_apo.to_csv(path + "/apo_pop.csv", index = False)

# =============================================================================
# population 변수 추가 
# =============================================================================
apo = pd.read_csv(path + '/apo_pop.csv')

# 2017
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 1), 'population'] = apo.loc[0, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 2), 'population'] = apo.loc[1, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 3), 'population'] = apo.loc[2, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 4), 'population'] = apo.loc[3, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 5), 'population'] = apo.loc[4, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 6), 'population'] = apo.loc[5, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 7), 'population'] = apo.loc[6, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 8), 'population'] = apo.loc[7, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 9), 'population'] = apo.loc[8, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 10), 'population'] = apo.loc[9, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 11), 'population'] = apo.loc[10, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 12), 'population'] = apo.loc[11, '아포읍 인구수']

# 2018
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 1), 'population'] = apo.loc[12, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 2), 'population'] = apo.loc[13, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 3), 'population'] = apo.loc[14, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 4), 'population'] = apo.loc[15, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 5), 'population'] = apo.loc[16, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 6), 'population'] = apo.loc[17, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 7), 'population'] = apo.loc[18, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 8), 'population'] = apo.loc[19, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 9), 'population'] = apo.loc[20, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 10), 'population'] = apo.loc[21, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 11), 'population'] = apo.loc[22, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 12), 'population'] = apo.loc[23, '아포읍 인구수']

# 2019
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 1), 'population'] = apo.loc[24, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 2), 'population'] = apo.loc[25, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 3), 'population'] = apo.loc[26, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 4), 'population'] = apo.loc[27, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 5), 'population'] = apo.loc[28, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 6), 'population'] = apo.loc[29, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 7), 'population'] = apo.loc[30, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 8), 'population'] = apo.loc[31, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 9), 'population'] = apo.loc[32, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 10), 'population'] = apo.loc[33, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 11), 'population'] = apo.loc[34, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 12), 'population'] = apo.loc[35, '아포읍 인구수']

# 2020
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 1), 'population'] = apo.loc[36, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 2), 'population'] = apo.loc[37, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 3), 'population'] = apo.loc[38, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 4), 'population'] = apo.loc[39, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 5), 'population'] = apo.loc[40, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 6), 'population'] = apo.loc[41, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 7), 'population'] = apo.loc[42, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 8), 'population'] = apo.loc[43, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 9), 'population'] = apo.loc[44, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 10), 'population'] = apo.loc[45, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 11), 'population'] = apo.loc[46, '아포읍 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 12), 'population'] = apo.loc[47, '아포읍 인구수']


# 2021(test)
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 1), 'population'] = apo.loc[48, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 2), 'population'] = apo.loc[49, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 3), 'population'] = apo.loc[50, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 4), 'population'] = apo.loc[51, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 5), 'population'] = apo.loc[52, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 6), 'population'] = apo.loc[53, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 7), 'population'] = apo.loc[54, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 8), 'population'] = apo.loc[55, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 9), 'population'] = apo.loc[56, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 10), 'population'] = apo.loc[57, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 11), 'population'] = apo.loc[58, '아포읍 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 12), 'population'] = apo.loc[59, '아포읍 인구수']

train['population'].shape # (35063,)
test['population'].shape # (8761,)
test.isnull().sum()

# 결측치 처리
# 2022년 1월 1일 00:00:00 결측치를 직전 데이터 값으로 대체
test.loc[8760, 'population'] = test.loc[8759, 'population']
test # 7800.0으로 결측치 채워짐

# =============================================================================
# 기후 변수 추가
# =============================================================================
### train
train_gc = pd.read_csv(path + '/train_gc_weather.csv') # train_gc_weather.csv를 가져와서 사용
train_gc
train_gc = train_gc.drop('풍속(m/s)', axis = 1)
train_gc.columns = ['ds','temp','rain','hum']
train_gc['ds'] = pd.to_datetime(train_gc['ds'])

# train에 기후 변수 추가(병합)
train = pd.merge(train, train_gc, how = 'outer', on = 'ds')
train.info()

### test
test_gc = pd.read_csv(path + '/test_gc_weather.csv')
test_gc
test_gc = test_gc.drop('풍속(m/s)', axis = 1)
test_gc.columns = ['ds','temp','rain','hum']
test_gc['ds'] = pd.to_datetime(test_gc['ds'])

# test에 기후 변수 추가(병합)
test = pd.merge(test, test_gc, how = 'outer', on = 'ds')
test.info()

# test set 추가 기간에 대한 기후 데이터 (2021-12-18 ~ 2022-01-01 00:00:00)
weather = pd.read_csv(path + '/test_gc_weather_added.csv')

weather.date = pd.to_datetime(weather.date)
weather = weather.drop('풍속(m/s)', axis = 1)
weather.columns = ['ds', 'temp', 'rain', 'hum']

for i in range(337) : # 337 = 들어갈 데이터 개수
  test.loc[8424 + i, 'temp'] = weather.loc[8392+i, 'temp'] # index번호로 매칭되어 있음

for i in range(337) :
  test.loc[8424 + i, 'rain'] = weather.loc[8392+i, 'rain']

for i in range(337) :
  test.loc[8424 + i, 'hum'] = weather.loc[8392+i, 'hum']

### 기후 데이터 이상치 확인 : 방재기상관측(AWS) 기준
'''
김천 기상시후 데이터 : 방재기상관측(AWS)
- AWS 상한/하한 기준
기온 : [-35, 45]
일강수량 : [0, 1500]
강수량 : 없음
강수유무 : [0, 10]
습도 : [0, 100]
'''

train[(train['temp'] < -35) | (train['temp'] > 45)] # 이상치 없음
train['temp'].describe()

test[(test['temp'] < -35) | (test['temp'] > 45)] # 이상치 없음
test['temp'].describe()

train[(train['rain'] < 0) | (train['rain'] > 300)] # 이상치 없음
train['rain'].describe()

test[(test['rain'] < 0) | (test['rain'] > 300)] # 이상치 없음
test['rain'].describe()

train[(train['hum'] < 0) | (train['hum'] > 100)] # 이상치 없음
train['hum'].describe()
# max        100.000000   <-- 상한값
train[(train['hum'] == 100)] # 1000 rows

test[(test['hum'] < 0) | (test['hum'] > 100)] # 이상치 없음
test['hum'].describe()

### 결측치 처리 (보간법으로 처리)
train.isnull().sum() 
'''
temp            223
rain            360
hum             267
'''
test.isnull().sum()
'''
temp             35
rain             54
hum              36
'''

# 기온은 시간의 영향을 많이 받을 것으로 예상되므로 time 옵션을 지정
train['hum'] = train['hum'].interpolate(option='linear')
train['rain'] = train['rain'].interpolate(option='linear')
train['temp'] = train['temp'].interpolate(option='time') # 기온 : time 옵션
train.isna().sum()

test['hum'] = test['hum'].interpolate(option='linear')
test['rain'] = test['rain'].interpolate(option='linear')
test['temp'] = test['temp'].interpolate(option='time') # 기온 : time 옵션
test.isna().sum()

# =============================================================================
# 체감온도(st) 변수 추가 (구미 데이터)
# =============================================================================
''' 메모장으로 파일 열어서
[검색조건]
지점명 : 구미, 기간 : 데이터 수집 기간
-> 문구 삭제하기
-> 인코딩 UTF-8로 변경 후 다른 이름으로 저장
'''

def add_st (data, new_data) :
    
    new_data.date = pd.to_datetime(new_data.date)
    data.ds = pd.to_datetime(data.ds)
    
    data['date'] = data['ds'].dt.date
    data.date = pd.to_datetime(data.date)
    
    data = pd.merge(data, new_data, on = 'date')
    data = data.drop('date', axis = 1)
    
    return data

### train data (20170101 ~ 20201231)
path2 = 'sensory_temperature' # 체감온도 데이터가 있는 path 설정

tr_st1 = pd.read_csv(path2 + '/체감온도_170101_170430.csv')
tr_st2 = pd.read_csv(path2 + '/체감온도_170501_170930.csv')
tr_st3 = pd.read_csv(path2 + '/체감온도_171001_180430.csv')
tr_st4 = pd.read_csv(path2 + '/체감온도_180501_180930.csv')
tr_st5 = pd.read_csv(path2 + '/체감온도_181001_190430.csv')
tr_st6 = pd.read_csv(path2 + '/체감온도_190501_190930.csv')
tr_st7 = pd.read_csv(path2 + '/체감온도_191001_200430.csv')
tr_st8 = pd.read_csv(path2 + '/체감온도_200501_200930.csv')
tr_st9 = pd.read_csv(path2 + '/체감온도_201001_201231.csv')

df_list = [tr_st1, tr_st2, tr_st3, tr_st4, tr_st5, tr_st6, tr_st7, tr_st8, tr_st9]

df = pd.concat(df_list, ignore_index = True)
df = df.iloc[:, [0,3]] # 필요한 열만 추출
df.columns = ['date', 'st']
df['date'] = pd.to_datetime(df['date'])
df

# 결측치 채우기(보간법)
df['st'].isnull().sum() # 결측치 94개
df['st'] = df['st'].interpolate(option = 'linear') 

df #(1461,2)

train = add_st(train, df)
train.isnull().sum()

### test data (20210101 ~ 20220101)
ts_st1 = pd.read_csv(path2 + '/체감온도_210101_210430.csv')
ts_st2 = pd.read_csv(path2 + '/체감온도_210501_210930.csv')
ts_st3 = pd.read_csv(path2 + '/체감온도_211001_220101.csv')

df2_list = [ts_st1, ts_st2, ts_st3]

df2 = pd.concat(df2_list, ignore_index = True)
df2 = df2.iloc[:, [0,3]]
df2.columns = ['date', 'st']
df2['date'] = pd.to_datetime(df2['date'])

# 결측치 채우기(보간법)
df2.isnull().sum() # 결측치 28개
df2['st'] = df2['st'].interpolate(option = 'linear') 

test = add_st(test, df2)
test.isnull().sum()      

# =============================================================================
# holiday 변수 추가 
# =============================================================================

# 휴일데이터 만드는 함수 / pytimekr을 이용한 휴일 추가
def holiday_feature(start, end, holiday2):
    hol = []
    for i in range(start,end+1):
        hol.append(pytimekr.holidays(i))

    day = []
    for i in range(0,end+1-start):
        for a in range(len(hol[i])):
            day.append(hol[i][a])
    holiday1 = pd.DataFrame(day)
    
    holiday2 = pd.DataFrame(holiday2)
    holiday_df = pd.concat([holiday1,holiday2],ignore_index=True)
    
    holiday_df = holiday_df.drop_duplicates() # 중복 날짜 제거
    holiday_df.columns = ['date'] # 컬럼이름 변환
    holiday_df['date'] = pd.to_datetime(holiday_df['date']) # 날짜 데이터로 변환
    
    # 공휴일중 주말 분리
    week = []
    for i in holiday_df.date.dt.weekday:
        if i == 5 or i == 6 : 
            week.append('0') # 주말 0
        else:
            week.append('1') # 주중 1
    holiday_df['holiday'] = week # 주중 주말 추가
    holiday_df = holiday_df[holiday_df['holiday'] == '1']
    holiday_df = holiday_df.sort_values('date')
     
    return holiday_df

# 대체휴일 추가
holiday2 = [dt.date(2017,1,30),dt.date(2017,5,9),dt.date(2017,10,2),dt.date(2017,10,6),
            dt.date(2018,5,7),dt.date(2018,9,26),
            dt.date(2019,5,6),
            dt.date(2020,1,27),dt.date(2020,8,17),
            dt.date(2021,8,16),dt.date(2021,10,4),dt.date(2021,10,11)]

## 2017~2022년 휴일 데이터
holiday_df = holiday_feature(2017, 2022, holiday2) #(시작 년도, 끝 년도, 대체휴일 데이터)

# 데이터를 합치는 함수
def plus_holiday(data,holiday_df):
    data['date'] = data['ds'].dt.date
    data['date'] = pd.to_datetime(data['date'])
    data_holi = pd.merge(data, holiday_df, how = 'left' ,on='date')
    data_holi = data_holi.drop('date',axis=1)
    data_holi['holiday'] = data_holi['holiday'].fillna(0)
    data_holi['holiday'] = data_holi['holiday'].astype('int64')

    return(data_holi)
 
# holiday 변수 추가 
train = plus_holiday(train,holiday_df)
test = plus_holiday(test, holiday_df)

########################################################################################
## 폭염 주의보 + 경보 / 경보, 주의보 기간을 직접 판단해 기간 삽입
########################################################################################

# Start, end를 train, test별로 구분
date_list_start_train = ['2017-05-19 10:00', '2017-05-28 16:00','2017-06-17 10:00','2017-06-20 10:30','2017-06-28 16:00','2017-07-03 11:30',
                        '2017-07-05 10:30','2017-07-09 11:00','2017-07-16 11:00','2017-07-26 10:00','2017-08-02 10:00','2017-08-11 16:00','2017-08-22 10:30',
                        '2018-06-01 16:00','2018-06-06 04:00','2018-06-22 04:00','2018-07-10 16:00','2018-07-24 11:00','2018-08-28 16:00',
                        '2019-05-22 16:00','2019-06-02 11:00','2019-06-04 11:00','2019-06-24 11:00','2019-07-02 11:00','2019-07-21 11:00','2019-07-27 11:00','2019-08-16 10:00',
                        '2019-09-09 13:00' ,'2020-06-03 11:00','2020-06-14 11:00','2020-06-21 10:30','2020-07-07 10:00','2020-07-20 11:00',
                        '2020-07-30 11:10','2020-08-08 11:00','2020-08-23 11:00']

date_list_start_test = ['2021-07-08 10:00','2021-07-19 10:00']

date_list_end_train = ['2017-05-20 16:00','2017-05-30 16:00','2017-06-19 18:00','2017-06-24 16:00','2017-07-01 10:00','2017-07-04 16:00',
                        '2017-07-06 16:00', '2017-07-15 09:00', '2017-07-24 16:00','2017-07-28 16:00','2017-08-08 16:00','2017-08-12 16:00','2017-08-25 16:00',
                        '2018-06-03 16:00','2018-06-08 16:00','2018-06-25 16:00','2018-07-15 14:00','2018-08-22 16:00','2018-08-30 04:00',
                        '2019-05-26 16:00','2019-06-03 11:00','2019-06-06 15:00','2019-06-25 16:00','2019-07-05 16:00','2019-07-24 16:00','2019-08-14 16:00','2019-08-19 16:00',
                        '2019-09-10 16:00','2020-06-12 16:00','2020-06-15 16:00','2020-06-23 16:00','2020-07-08 18:20','2020-07-21 16:00',
                        '2020-08-05 16:00','2020-08-21 16:00','2020-08-31 16:30']

date_list_end_test =['2021-07-16 17:00','2021-08-12 17:00']

# Series + datetime 지정
date_list_start_train = pd.Series(date_list_start_train)
date_list_start_train = pd.to_datetime(date_list_start_train)

date_list_start_test = pd.Series(date_list_start_test)
date_list_start_test = pd.to_datetime(date_list_start_test)

date_list_end_train = pd.Series(date_list_end_train)
date_list_end_train = pd.to_datetime(date_list_end_train)

date_list_end_test = pd.Series(date_list_end_test)
date_list_end_test = pd.to_datetime(date_list_end_test)

# start ~ end 사이의 구간 만들기
date_train = []
for start, end in zip(date_list_start_train, date_list_end_train) :
    rr = pd.date_range(start, end, freq='1H')
    date_train.append(rr)

date_test = []
for start, end in zip(date_list_start_test, date_list_end_test) :
    bb = pd.date_range(start, end, freq='1H')
    date_test.append(bb)

# 경보, 주의보 발령 기간 생성 + 발령 여부를 warning으로 표시
sample_tr = pd.DataFrame([])
for iter in range(len(date_train)) :
    sample_tr = pd.concat([sample_tr, pd.Series(date_train[iter])], axis=0)

sample_ts = pd.DataFrame([])
for iter in range(len(date_test)) :
    sample_ts = pd.concat([sample_ts, pd.Series(date_test[iter])], axis=0)

sample_tr['warning'] = 'warning'
sample_tr.columns = ['ds','warning']

sample_ts['warning'] = 'warning'
sample_ts.columns = ['ds','warning']

# 전체 데이터셋 기간에 대해 위 경보, 주의보 기간을 삽입
all_tr = pd.DataFrame(pd.date_range('2017-01-01 01:00:00','2020-12-31 23:00:00', freq='1H'))
all_tr['warning'] = 'not'
all_tr.columns = ['ds','warning']

all_ts = pd.DataFrame(pd.date_range('2021-01-01 00:00:00','2022-01-01 00:00:00', freq='1H'))
all_ts['warning'] = 'not'
all_ts.columns = ['ds','warning']

comp_tr = pd.merge(all_tr, sample_tr, on='ds', how='left')
comp_tr.drop('warning_x',axis=1,inplace=True)
comp_tr.warning_y.fillna('not', inplace=True)

comp_ts = pd.merge(all_ts, sample_ts, on='ds', how='left')
comp_ts.drop('warning_x',axis=1,inplace=True)
comp_ts.warning_y.fillna('not', inplace=True)

train['warning'] = comp_tr.warning_y
test['warning'] = comp_ts.warning_y

train.warning.value_counts()
test.warning.value_counts()

#########################################################################################
## 자체 평가지표 계산을 위해 y_test NaN 채우기 -> 작년 지표 활용
########################################################################################

new = test.y.dropna()
new2 = train.loc[(train.ds >= '2020-12-18 00:00:00') & (train.ds <= '2021-12-31 23:00:00'),'y']
new3 = test.loc[(test.ds == '2021-01-01 00:00:00'), 'y']
new_col = pd.concat([new, new2, new3],axis=0)

test.drop('y', axis=1, inplace=True)
new_col = new_col.reset_index()
new_col.drop('index', inplace=True, axis=1)

test = pd.concat([test, new_col], axis=1)

train.to_csv('dataset/task2_sub_final_tr.csv')
test.to_csv('dataset/task2_sub_final_ts.csv')

#####################
## data import
#####################
sample_df = pd.read_csv('dataset/data_city/sample_city.csv',index_col=0)
train_df = pd.read_csv('dataset/task2_sub_final_tr.csv', index_col=0)
test_df_xgb = pd.read_csv('dataset/task2_sub_final_ts.csv', index_col=0)

test_df_xgb.drop('index',axis=1,inplace=True)

def season_calc(month):
    if month in [5,6,7,8]:
        return "summer"
    else:
        return "winter"

def is_season(ds) :
    date = pd.to_datetime(ds)
    return (date.month > 10 or date.month < 4)

def feature_preprocessing(df) :
    # 'ds' 데이터형 변경
    df.ds = pd.to_datetime(df['ds'])

    # ds 기반 새로운 feature 생성
    df['month'] = df.ds.dt.month
    df['year'] = df.ds.dt.year
    df['day'] = df.ds.dt.day
    df['hour'] = df.ds.dt.hour
    weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
    df['weekday'] = df.ds.dt.weekday.map(weekdays)

    df['dayofyear'] = df.ds.dt.dayofyear

    # 계절에 관한 feature 생성
    for iter in range(len(df)) :
        df.loc[iter,'season'] = season_calc(df.loc[iter, 'month'])

    # hour 변환
    df['time_of_day'] = df['hour'].apply(time_of_day)
    df['time'] = df['hour'].apply(time)

    # dtype 변경
    cat_cols = ['time_of_day','season','time','weekday','warning', 'holiday']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    df['population'] = df['population'].astype(float)
    df['population'] = df['population'].astype(int)

    # get_dummies
    df = pd.get_dummies(df, drop_first=True)

    return df

####################
## outlier + missing value
####################

# 결측값 개수 확인
train_df.isna().sum() # 'y' : 8
test_df_xgb.isna().sum()

# outlier를 missing value로 대체
train_df.loc[(abs(train_df['y']) > 2000) | (train_df['y'] == 0), 'y'] = np.NaN # 이상치를 missing value로 전환 -> 153개의 missing value 생성
train_df['y'].isna().sum()

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
    elif method == 'time' :
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

# Interpolation - polynomial

train_df_Interpolation_poly = missing_value_func(train_df,'polynomial')

train_df_Interpolation_time = feature_preprocessing(train_df_Interpolation_time) # 변수 전처리 함수
test_df_xgb = feature_preprocessing(test_df_xgb)

train_df_Interpolation_time.set_index('ds',inplace=True)
test_df_xgb.set_index('ds',inplace=True)

train_df_Interpolation_time.columns.values
test_df_xgb.columns.values

#########################################################################################
## Model fitting
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

## lag variable 생성
for i in range(24):
    train_df_final['lag'+str(i+1)] = train_df_final['y'].shift(i+1)

for i in range(24):
    test_df_final['lag'+str(i+1)] = test_df_final['y'].shift(i+1)

# 불필요한 컬럼 삭제
train_df_lag = train_df_final.drop(['day','rain','time_evening','time_morning','time_night','dayofyear','weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
       'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday'], axis=1)
train_df_lag = train_df_lag.dropna()
test_df_lag = test_df_final.drop(['day','rain','time_evening','time_morning','time_night','dayofyear','weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
       'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday'], axis=1)

# fourier항 추가
test_df_lag.index = pd.to_datetime(test_df_lag.index)
train_df_lag = add_fourier_terms(train_df_lag, year_k=11, week_k = 12, day_k =12)
test_df_lag = add_fourier_terms(test_df_lag, year_k= 11, week_k=12, day_k=12)

# rolling feature 추가
train_df_mul = add_feature(train_df_lag)
test_df_mul = add_feature(test_df_lag)
train_df_mul.dropna(inplace=True)

Best_params = {'subsample': 1.0, 'n_estimators': 73, 'min_child_weight': 0.5, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.25, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.8}
xgbtuned = xgb.XGBRegressor(**Best_params)

X_train = train_df_mul.drop('y', axis=1)
y_train = train_df_mul.y

X_test = test_df_mul.drop('y', axis=1)
y_test = test_df_mul.y

xgbtuned.fit(X_train, y_train)
pred = xgbtuned.predict(X_test)

mae = mean_absolute_error(y_test,pred)
print(mae)

#joblib.dump(xgbtuned, 'result/task2_submit_model_final.pkl')

"""
# 제출용 csv파일 생성 함수
result = []
for iter in range(len(X_test)-336) :

    mid = pred[0+iter:336+iter]
    result.append(mid)

result = np.array(result)
result = result.squeeze()
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns
"""
# result.to_csv('task2_submit_pred_final.csv')