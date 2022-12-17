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

def add_hour_fourier(df, day_k) :

    
    for k in range(1, day_k+1):
        
        # day has period of 24
        df['hour_sin'+str(k)] = np.sin(2 *k* np.pi * df.index.hour/24)
        df['hour_cos'+str(k)] = np.cos(2 *k* np.pi * df.index.hour/24)

    return df

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
path = 'dataset/data_city'
a = pd.read_csv(path + '/data_tr_city.csv')
b = pd.read_csv(path + '/data_ts_city.csv')

train = a.copy()
test = b.copy()

train.columns = ['ds', 'y']
test.columns = ['ds', 'y']

train.ds = pd.to_datetime(train.ds)
test.ds = pd.to_datetime(test.ds)

# =============================================================================
#  test set 기간 추가 ( ~ 2022.01.01 00:00:00)
# =============================================================================
s = pd.Series(pd.date_range("2021-12-18 00:00:00", periods=337, freq = '0D1H0min0S'))

df_s = pd.DataFrame({'ds' : s})

test = pd.concat([test, df_s])
test = test.reset_index()
test

# =============================================================================
# week_weekday 변수 추가
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
import numpy as np
import pandas as pd

# 인구수 율곡동
# Dataset Import
kim2017 = pd.read_csv('population/population_2017.csv', encoding='cp949', thousands=',')
kim2018 = pd.read_csv('population/population_2018.csv', encoding='cp949', thousands=',')
kim2019 = pd.read_csv('population/population_2019.csv', encoding='cp949', thousands=',')
kim2020 = pd.read_csv('population/population_2020.csv', encoding='cp949', thousands=',')
kim2021 = pd.read_csv('population/population_2021.csv', encoding='cp949', thousands=',')
combine = [kim2017, kim2018, kim2019, kim2020, kim2021]

# DataFrame
year_2017=pd.DataFrame(kim2017)
year_2018=pd.DataFrame(kim2018)
year_2019=pd.DataFrame(kim2019)
year_2020=pd.DataFrame(kim2020)
year_2021=pd.DataFrame(kim2021)

final_yul2017=year_2017[['2017년01월_총인구수','2017년02월_총인구수','2017년03월_총인구수','2017년04월_총인구수',
             '2017년05월_총인구수','2017년06월_총인구수','2017년07월_총인구수','2017년08월_총인구수',
             '2017년09월_총인구수','2017년10월_총인구수','2017년11월_총인구수','2017년12월_총인구수']]

final_yul2018=year_2018[['2018년01월_총인구수','2018년02월_총인구수','2018년03월_총인구수','2018년04월_총인구수',
             '2018년05월_총인구수','2018년06월_총인구수','2018년07월_총인구수','2018년08월_총인구수',
             '2018년09월_총인구수','2018년10월_총인구수','2018년11월_총인구수','2018년12월_총인구수']]

final_yul2019=year_2019[['2019년01월_총인구수','2019년02월_총인구수','2019년03월_총인구수','2019년04월_총인구수',
             '2019년05월_총인구수','2019년06월_총인구수','2019년07월_총인구수','2019년08월_총인구수',
             '2019년09월_총인구수','2019년10월_총인구수','2019년11월_총인구수','2019년12월_총인구수']]

final_yul2020=year_2020[['2020년01월_총인구수','2020년02월_총인구수','2020년03월_총인구수','2020년04월_총인구수',
             '2020년05월_총인구수','2020년06월_총인구수','2020년07월_총인구수','2020년08월_총인구수',
             '2020년09월_총인구수','2020년10월_총인구수','2020년11월_총인구수','2020년12월_총인구수']]

final_yul2021=year_2021[['2021년01월_총인구수','2021년02월_총인구수','2021년03월_총인구수','2021년04월_총인구수',
             '2021년05월_총인구수','2021년06월_총인구수','2021년07월_총인구수','2021년08월_총인구수',
             '2021년09월_총인구수','2021년10월_총인구수','2021년11월_총인구수','2021년12월_총인구수']]
final_yul2017.T

# 율곡동의 행 순번 22 -> 22행 추출
final_yul2017=final_yul2017.loc[[22]]
final_yul2018=final_yul2018.loc[[22]]
final_yul2019=final_yul2019.loc[[22]]
final_yul2020=final_yul2020.loc[[22]]
final_yul2021=final_yul2021.loc[[22]]

type(final_yul2017)

yulgok17=final_yul2017.rename(columns={'2017년01월_총인구수':'1월',
                             '2017년02월_총인구수':'2월',
                             '2017년03월_총인구수':'3월',
                             '2017년04월_총인구수':'4월',
                             '2017년05월_총인구수':'5월',
                             '2017년06월_총인구수':'6월',
                             '2017년07월_총인구수':'7월',
                             '2017년08월_총인구수':'8월',
                             '2017년09월_총인구수':'9월',
                             '2017년10월_총인구수':'10월',
                             '2017년11월_총인구수':'11월',
                             '2017년12월_총인구수':'12월'},index={22:'율곡동 2017년'})

yulgok18=final_yul2018.rename(columns={'2018년01월_총인구수':'1월',
                             '2018년02월_총인구수':'2월',
                             '2018년03월_총인구수':'3월',
                             '2018년04월_총인구수':'4월',
                             '2018년05월_총인구수':'5월',
                             '2018년06월_총인구수':'6월',
                             '2018년07월_총인구수':'7월',
                             '2018년08월_총인구수':'8월',
                             '2018년09월_총인구수':'9월',
                             '2018년10월_총인구수':'10월',
                             '2018년11월_총인구수':'11월',
                             '2018년12월_총인구수':'12월'},index={22:'율곡동 2018년'})
yulgok19=final_yul2019.rename(columns={'2019년01월_총인구수':'1월',
                             '2019년02월_총인구수':'2월',
                             '2019년03월_총인구수':'3월',
                             '2019년04월_총인구수':'4월',
                             '2019년05월_총인구수':'5월',
                             '2019년06월_총인구수':'6월',
                             '2019년07월_총인구수':'7월',
                             '2019년08월_총인구수':'8월',
                             '2019년09월_총인구수':'9월',
                             '2019년10월_총인구수':'10월',
                             '2019년11월_총인구수':'11월',
                             '2019년12월_총인구수':'12월'},index={22:'율곡동 2019년'})
yulgok20=final_yul2020.rename(columns={'2020년01월_총인구수':'1월',
                             '2020년02월_총인구수':'2월',
                             '2020년03월_총인구수':'3월',
                             '2020년04월_총인구수':'4월',
                             '2020년05월_총인구수':'5월',
                             '2020년06월_총인구수':'6월',
                             '2020년07월_총인구수':'7월',
                             '2020년08월_총인구수':'8월',
                             '2020년09월_총인구수':'9월',
                             '2020년10월_총인구수':'10월',
                             '2020년11월_총인구수':'11월',
                             '2020년12월_총인구수':'12월'},index={22:'율곡동 2020년'})
yulgok21=final_yul2021.rename(columns={'2021년01월_총인구수':'1월',
                             '2021년02월_총인구수':'2월',
                             '2021년03월_총인구수':'3월',
                             '2021년04월_총인구수':'4월',
                             '2021년05월_총인구수':'5월',
                             '2021년06월_총인구수':'6월',
                             '2021년07월_총인구수':'7월',
                             '2021년08월_총인구수':'8월',
                             '2021년09월_총인구수':'9월',
                             '2021년10월_총인구수':'10월',
                             '2021년11월_총인구수':'11월',
                             '2021년12월_총인구수':'12월'},index={22:'율곡동 2021년'})

new_2017=final_yul2017.rename(index={22:'율곡동 인구수'})
new_2018=final_yul2018.rename(index={22:'율곡동 인구수'})
new_2019=final_yul2019.rename(index={22:'율곡동 인구수'})
new_2020=final_yul2020.rename(index={22:'율곡동 인구수'})
new_2021=final_yul2021.rename(index={22:'율곡동 인구수'})

new_2017=new_2017.T
new_2018=new_2018.T
new_2019=new_2019.T
new_2020=new_2020.T
new_2021=new_2021.T

fin_gok =pd.concat([new_2017,new_2018,new_2019,new_2020,new_2021],axis=0)

fin_gok.to_csv(path + "/yulgok_pop.csv", index = False)

# =============================================================================
# population 변수 추가 
# =============================================================================
yulgok = pd.read_csv(path + '/yulgok_pop.csv')

# 2017
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 1), 'population'] = yulgok.loc[0, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 2), 'population'] = yulgok.loc[1, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 3), 'population'] = yulgok.loc[2, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 4), 'population'] = yulgok.loc[3, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 5), 'population'] = yulgok.loc[4, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 6), 'population'] = yulgok.loc[5, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 7), 'population'] = yulgok.loc[6, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 8), 'population'] = yulgok.loc[7, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 9), 'population'] = yulgok.loc[8, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 10), 'population'] = yulgok.loc[9, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 11), 'population'] = yulgok.loc[10, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 12), 'population'] = yulgok.loc[11, '율곡동 인구수']

# 2018
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 1), 'population'] = yulgok.loc[12, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 2), 'population'] = yulgok.loc[13, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 3), 'population'] = yulgok.loc[14, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 4), 'population'] = yulgok.loc[15, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 5), 'population'] = yulgok.loc[16, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 6), 'population'] = yulgok.loc[17, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 7), 'population'] = yulgok.loc[18, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 8), 'population'] = yulgok.loc[19, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 9), 'population'] = yulgok.loc[20, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 10), 'population'] = yulgok.loc[21, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 11), 'population'] = yulgok.loc[22, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 12), 'population'] = yulgok.loc[23, '율곡동 인구수']

# 2019
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 1), 'population'] = yulgok.loc[24, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 2), 'population'] = yulgok.loc[25, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 3), 'population'] = yulgok.loc[26, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 4), 'population'] = yulgok.loc[27, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 5), 'population'] = yulgok.loc[28, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 6), 'population'] = yulgok.loc[29, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 7), 'population'] = yulgok.loc[30, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 8), 'population'] = yulgok.loc[31, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 9), 'population'] = yulgok.loc[32, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 10), 'population'] = yulgok.loc[33, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 11), 'population'] = yulgok.loc[34, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 12), 'population'] = yulgok.loc[35, '율곡동 인구수']

# 2020
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 1), 'population'] = yulgok.loc[36, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 2), 'population'] = yulgok.loc[37, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 3), 'population'] = yulgok.loc[38, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 4), 'population'] = yulgok.loc[39, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 5), 'population'] = yulgok.loc[40, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 6), 'population'] = yulgok.loc[41, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 7), 'population'] = yulgok.loc[42, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 8), 'population'] = yulgok.loc[43, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 9), 'population'] = yulgok.loc[44, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 10), 'population'] = yulgok.loc[45, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 11), 'population'] = yulgok.loc[46, '율곡동 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 12), 'population'] = yulgok.loc[47, '율곡동 인구수']

# 2021(test)
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 1), 'population'] = yulgok.loc[48, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 2), 'population'] = yulgok.loc[49, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 3), 'population'] = yulgok.loc[50, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 4), 'population'] = yulgok.loc[51, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 5), 'population'] = yulgok.loc[52, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 6), 'population'] = yulgok.loc[53, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 7), 'population'] = yulgok.loc[54, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 8), 'population'] = yulgok.loc[55, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 9), 'population'] = yulgok.loc[56, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 10), 'population'] = yulgok.loc[57, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 11), 'population'] = yulgok.loc[58, '율곡동 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 12), 'population'] = yulgok.loc[59, '율곡동 인구수']

train['population'].shape # (35063,)
test['population'].shape # (8761,)
train.isnull().sum()              
test.isnull().sum()

# 결측치 처리
# 2022년 1월 1일 00:00:00 결측치를 직전 데이터 값으로 대체
test.loc[8760, 'population'] = test.loc[8759, 'population']
test # 22894.0으로 결측치 채워짐

# =============================================================================
# 기후 변수 추가
# =============================================================================
### train
train_gc = pd.read_csv(path + '/train_gc_weather.csv')
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

test.drop('index', axis=1, inplace=True)

# test set 추가 기간에 대한 기후 데이터
weather = pd.read_csv(path + '/test_gc_weather_added.csv')

weather.date = pd.to_datetime(weather.date)
weather = weather.drop('풍속(m/s)', axis = 1)
weather.columns = ['ds', 'temp', 'rain', 'hum']

for i in range(337) :
  test.loc[8424 + i, 'temp'] = weather.loc[8392+i, 'temp']

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
def add_st (data, new_data) :
    
    new_data.date = pd.to_datetime(new_data.date)
    data.ds = pd.to_datetime(data.ds)
    
    data['date'] = data['ds'].dt.date
    data.date = pd.to_datetime(data.date)
    
    data = pd.merge(data, new_data, on = 'date')
    data = data.drop('date', axis = 1)
    
    return data

### train data (20170101 ~ 20201231)
path2 = 'sensory_temperature'

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
df = df.iloc[:, [0,3]]
df.columns = ['date', 'st']
df['date'] = pd.to_datetime(df['date'])

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

# 휴일데이터 만드는 함수
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
    data_holi['holiday'] = data_holi['holiday'].astype(object)

    return(data_holi)
 
# holiday 변수 추가 
train = plus_holiday(train,holiday_df)
test = plus_holiday(test, holiday_df)

########################################################################################
## 폭염 주의보 + 경보
########################################################################################

"""
2017-05-19 10:00 ~ 2017-05-20 16:00
2017-05-28 16:00 ~ 2017-05-30 16:00
2017-06-17 10:00 ~ 2017-06-19 18:00
2017-06-20 10:30 ~ 2017-06-24 16:00
2017-06-28 16:00 ~ 2017-07-01 10:00
2017-07-03 11:30 ~ 2017-07-04 16:00
2017-07-05 10:30 ~ 2017-07-06 16:00
2017-07-09 11:00 ~ 2017-07-15 09:00
2017-07-16 11:00 ~ 2017-07-24 16:00
2017-07-26 10:00 ~ 2017-07-28 16:00
2017-08-02 10:00 ~ 2017-08-08 16:00
2017-08-11 16:00 ~ 2017-08-12 16:00
2017-08-22 10:30 ~ 2017-08-25 16:00

2018-06-01 16:00 ~ 2018-06-03 16:00
2018-06-06 04:00 ~ 2018-06-08 16:00
2018-06-22 04:00 ~ 2018-06-25 16:00
2018-07-10 16:00 ~ 2018-07-15 14:00
2018-07-24 11:00 ~ 2018-08-22 16:00
2018-08-28 16:00 ~ 2018-08-30 04:00

2019-05-22 16:00 ~ 2019-05-26 16:00
2019-06-02 11:00 ~ 2019-06-03 11:00
2019-06-04 11:00 ~ 2019-06-06 15:00
2019-06-24 11:00 ~ 2019-06-25 16:00
2019-07-02 11:00 ~ 2019-07-05 16:00
2019-07-21 11:00 ~ 2019-07-24 16:00
2019-07-27 11:00 ~ 2019-08-14 16:00
2019-08-16 10:00 ~ 2019-08-19 16:00
2019-09-09 13:00 ~  2019-09-10 16:00

2020-06-03 11:00 ~ 2020-06-12 16:00
2020-06-14 11:00 ~ 2020-06-15 16:00
2020-06-21 10:30 ~ 2020-06-23 16:00
2020-07-07 10:00 ~ 2020-07-08 18:20
2020-07-20 11:00 ~ 2020-07-21 16:00
2020-07-30 11:10 ~ 2020-08-05 16:00
2020-08-08 11:00 ~ 2020-08-21 16:00
2020-08-23 11:00 ~ 2020-08-31 16:30

2021-07-08 10:00 ~ 2021-07-16 17:00
2021-07-19 10:00 ~ 2021-08-12 17:00
"""

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

date_list_start_train = pd.Series(date_list_start_train)
date_list_start_train = pd.to_datetime(date_list_start_train)

date_list_start_test = pd.Series(date_list_start_test)
date_list_start_test = pd.to_datetime(date_list_start_test)

date_list_end_train = pd.Series(date_list_end_train)
date_list_end_train = pd.to_datetime(date_list_end_train)

date_list_end_test = pd.Series(date_list_end_test)
date_list_end_test = pd.to_datetime(date_list_end_test)

date_train = []
for start, end in zip(date_list_start_train, date_list_end_train) :
    rr = pd.date_range(start, end, freq='1H')
    date_train.append(rr)

date_test = []
for start, end in zip(date_list_start_test, date_list_end_test) :
    bb = pd.date_range(start, end, freq='1H')
    date_test.append(bb)

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

train.to_csv('dataset/task1_sub_final_tr.csv')
test.to_csv('dataset/task1_sub_final_ts.csv')

#####################
## data import
#####################
train_df = pd.read_csv('dataset/data_city/task1_city_tr_data.csv',index_col=0)
sample_df = pd.read_csv('dataset/data_city/sample_city.csv',index_col=0)
test_df_xgb = pd.read_csv('dataset/data_city/city_ts_final.csv',index_col=0)

train_df_2 = pd.read_csv('dataset/task1_sub_final_tr.csv', index_col=0)
test_df_xgb_2 = pd.read_csv('dataset/task1_sub_final_ts.csv', index_col=0)

train_df.drop('index',axis=1,inplace=True)
test_df_xgb.drop(['hour','year','month','day'], axis=1, inplace=True)

bbb = np.equal(np.array(train_df.y), np.array(train_df_2.y))
aaa = np.equal(np.array(test_df_xgb.y), np.array(test_df_xgb_2.y))

b45 = pd.read_csv(path + '/data_ts_city.csv')
b45.columns = ['ds','y']

idx = []
for iter in range(len(aaa)) :
    if aaa[iter] == False :
        idx.append(iter)

idx2 = []
for iter in range(len(bbb)) :
    if bbb[iter] == False :
        idx2.append(iter)

s1 = test_df_xgb.loc[idx,'y']
s2 = test_df_xgb_2.loc[idx, 'y']
s4 = b45.loc[idx,'y']

t1 = train_df.loc[idx2,'y']
t2 = train_df_2.loc[idx2,'y']

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

train_df.columns
test_df_xgb.columns

####################
## outlier + missing value
####################

# 결측값 개수 확인
train_df.isna().sum() # 'y' : 8
test_df_xgb.isna().sum()

# outlier를 missing value로 대체
train_df.loc[(abs(train_df['y']) > 2000) | (train_df['y'] == 0), 'y'] = np.NaN # 이상치를 missing value로 전환 -> 153개의 missing value 생성
train_df['y'].isna().sum()

sum(np.equal(np.array(train_df.y), np.array(train_df_2.y)) == False)

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

train_df_Interpolation_time = feature_preprocessing(train_df_Interpolation_time)
test_df_xgb = feature_preprocessing(test_df_xgb)

train_df_Interpolation_time.set_index('ds',inplace=True)
test_df_xgb.set_index('ds',inplace=True)

train_df_Interpolation_time_2 = missing_value_func(train_df_2, 'time')

train_df_Interpolation_time_2 = feature_preprocessing(train_df_Interpolation_time_2)
test_df_xgb_2 = feature_preprocessing(test_df_xgb_2)

train_df_Interpolation_time_2.set_index('ds',inplace=True)
test_df_xgb_2.set_index('ds',inplace=True)

train_df_Interpolation_time.columns.values

sum(np.equal(np.array(train_df_Interpolation_time.y), np.array(train_df_Interpolation_time_2.y)) == False) # False

sum(np.equal(np.array(test_df_xgb.y), np.array(test_df_xgb_2.y)) == False)

########################################################################################
## Cross Validation
#########################################################################################

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# hyperparameter 설정
param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        #'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        'learning_rate' : [0.01, 0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.15,0.2],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'n_estimators': [10, 31, 52, 73, 94, 115, 136, 157, 178, 200]}

#########################################################################################
## RandomizedSearchCV
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

train_df_lag = add_fourier_terms(train_df_lag, year_k= 12, week_k = 12, day_k =12)
test_df_lag = add_fourier_terms(test_df_lag, year_k= 12, week_k=12, day_k=12)

# rolling feature 추가
train_df_mul = add_feature(train_df_lag)
test_df_mul = add_feature(test_df_lag)
train_df_mul.dropna(inplace=True)

################################
# best model
################################

xgbtuned = xgb.XGBRegressor(subsample=0.9, n_estimators=115, min_child_weight=7.0, max_depth=6, learning_rate=0.045, gamma=0.25, colsample_bytree=0.9,
                            colsample_bylevel=0.5)                           

X_train = train_df_mul.drop('y', axis=1)
y_train = train_df_mul.y

X_test = test_df_mul.drop('y', axis=1)
y_test = test_df_mul.y

xgbtuned.fit(X_train, y_train)
preds_y = xgbtuned.predict(X_test)
mae = mean_absolute_error(y_test, preds_y)
print(mae)

from sklearn.model_selection import GridSearchCV
param = {'learning_rate' : [0.04  , 0.0401, 0.0402, 0.0403, 0.0404, 0.0405, 0.0406, 0.0407,
       0.0408, 0.0409, 0.041 , 0.0411, 0.0412, 0.0413, 0.0414, 0.0415,
       0.0416, 0.0417, 0.0418, 0.0419, 0.042 , 0.0421, 0.0422, 0.0423,
       0.0424, 0.0425, 0.0426, 0.0427, 0.0428, 0.0429, 0.043 , 0.0431,
       0.0432, 0.0433, 0.0434, 0.0435, 0.0436, 0.0437, 0.0438, 0.0439,
       0.044 , 0.0441, 0.0442, 0.0443, 0.0444, 0.0445, 0.0446, 0.0447,
       0.0448, 0.0449, 0.045 , 0.0451, 0.0452, 0.0453, 0.0454, 0.0455,
       0.0456, 0.0457, 0.0458, 0.0459, 0.046 , 0.0461, 0.0462, 0.0463,
       0.0464, 0.0465, 0.0466, 0.0467, 0.0468, 0.0469, 0.047 , 0.0471,
       0.0472, 0.0473, 0.0474, 0.0475, 0.0476, 0.0477, 0.0478, 0.0479,
       0.048 , 0.0481, 0.0482, 0.0483, 0.0484, 0.0485, 0.0486, 0.0487,
       0.0496, 0.0497, 0.0498, 0.0499, 0.05  , 0.0501, 0.0502, 0.0503,
       0.0504, 0.0505, 0.0506, 0.0507, 0.0508, 0.0509, 0.051 , 0.0511,
       0.0512, 0.0513, 0.0514, 0.0515, 0.0516, 0.0517, 0.0518, 0.0519,
       0.052 , 0.0521, 0.0522, 0.0523, 0.0524, 0.0525, 0.0526, 0.0527,
       0.0528, 0.0529, 0.053 , 0.0531, 0.0532, 0.0533, 0.0534, 0.0535,
       0.0536, 0.0537, 0.0538, 0.0539, 0.054 , 0.0541, 0.0542, 0.0543,
       0.0544, 0.0545, 0.0546, 0.0547, 0.0548, 0.0549, 0.055 , 0.0551,
       0.0552, 0.0553, 0.0554, 0.0555, 0.0556, 0.0557, 0.0558, 0.0559,
       0.056 , 0.0561, 0.0562, 0.0563, 0.0564, 0.0565, 0.0566, 0.0567,
       0.0568, 0.0569, 0.057 , 0.0571, 0.0572, 0.0573, 0.0574, 0.0575,
       0.0576, 0.0577, 0.0578, 0.0579, 0.058 , 0.0581, 0.0582, 0.0583,
       0.0584, 0.0585, 0.0586, 0.0587, 0.0588, 0.0589, 0.059 , 0.0591,
       0.0592, 0.0593, 0.0594, 0.0595, 0.0596, 0.0597, 0.0598, 0.0599]}

xgb = xgb.XGBRegressor(subsample=0.9, n_estimators=115, min_child_weight=7.0, max_depth=6, gamma=0.25, colsample_bytree=0.9,
                            colsample_bylevel=0.5, tree_method='gpu_hist')
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(xgb, param_grid=param, verbose=2, n_jobs=-1,cv=tscv)
grid.fit(X_train, y_train)

preds_y = grid.predict(X_test)
mae = mean_absolute_error(y_test, preds_y)
print(mae)

# joblib.dump(xgbtuned, 'result/best_model_now.pkl')

np.arange(0.04,0.06,0.0001)

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

# result.to_csv('result/xgb_hyper_comp.csv')

#########################################################################################
## 결과 제출
#########################################################################################

train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

# Hyperparameter 설정
param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07,0.08,0.09,0.1,0.11,
                        0.12,0.13,0.14,0.15,0.16],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'n_estimators': [10, 31, 52, 73, 94, 115, 136, 157, 178, 200]}

# 결과 생성 함수
def make_result(train_df, test_df, param_grid, n_iter, batch_size) :

    ## lag variable 생성
    for i in range(24):
        train_df['lag'+str(i+1)] = train_df['y'].shift(i+1)

    for i in range(24):
        test_df['lag'+str(i+1)] = test_df['y'].shift(i+1)

    # 제외할 변수 선택
    train_df_lag = train_df.drop(['day','rain','time_evening','time_morning','time_night','dayofyear','weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
        'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday'], axis=1)
    train_df_lag = train_df_lag.dropna() # lag로 인한 NaN 제거
    test_df_lag = test_df.drop(['day','rain','time_evening','time_morning','time_night','dayofyear','weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
        'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday'], axis=1)

    # fourier항 생성
    test_df_lag.index = pd.to_datetime(test_df_lag.index)
    train_df_lag = add_fourier_terms(train_df_lag, year_k= 12, week_k=12, day_k=12)
    test_df_lag = add_fourier_terms(test_df_lag, year_k= 12, week_k=12, day_k=12)

    # rolling feature 생성
    train_df_mul = add_feature(train_df_lag)
    test_df_mul = add_feature(test_df_lag)
    train_df_mul.dropna(inplace=True)

    # train/test data split
    X_train = train_df_mul.drop('y', axis=1)
    y_train = train_df_mul.y

    X_test = test_df_mul.drop('y', axis=1)
    y_test = test_df_mul.y

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # model 생성 / 조절 가능 parameter = n_splits, n_iter
    xgbtuned = xgb.XGBRegressor(tree_method='gpu_hist',random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    xgbtunedreg = RandomizedSearchCV(xgbtuned, param_distributions=param_grid , 
                                    scoring='neg_mean_absolute_error', n_iter=n_iter, n_jobs=-1, 
                                    cv=tscv, verbose=2, random_state=42)
    
    # model 데이터 생성 / batch_size 조절 가능, joblib으로 생성되는 파일 이름 변경 / return으로 pred_y가 생성되고 이 mae값이 최소라면 하단의 result 생성함수를 이용해 결과를 생성
    pred_y = pd.DataFrame([])
    batch_size = batch_size
    best_params = []
        
    for iter in range(len(X_test)//batch_size) :
        if iter == 0 :
            test = X_test[:batch_size*(1+iter)]

            xgbtunedreg.fit(X_train, y_train, verbose=True)
            preds= xgbtunedreg.predict(test)
            pred_y = pd.concat([pred_y, pd.Series(preds)],axis=0)

            best_params.append(xgbtunedreg.best_params_)
            joblib.dump(xgbtunedreg, 'result/task1_xgb_model_'+str(iter)+'.pkl')

        elif iter == (len(X_test)//batch_size)-1 :
            X_train_add = X_test[(iter-1)*batch_size:iter*batch_size]
            y_train_add = y_test[(iter-1)*batch_size:iter*batch_size]
            X_train = pd.concat([X_train, X_train_add], axis=0)
            y_train = pd.concat([y_train, y_train_add], axis=0)

            test = X_test[(iter)*batch_size:]

            xgbtunedreg.fit(X_train, y_train, verbose=True)
            preds = xgbtunedreg.predict(test)
            pred_y = pd.concat([pred_y, pd.Series(preds)],axis=0)

            best_params.append(xgbtunedreg.best_params_)

            joblib.dump(xgbtunedreg, 'result/task1_xgb_model_'+str(iter)+'.pkl')
            print('now step =', iter)

        else :
            X_train_add = X_test[(iter-1)*batch_size:iter*batch_size]
            y_train_add = y_test[(iter-1)*batch_size:iter*batch_size]
            X_train = pd.concat([X_train, X_train_add], axis=0)
            y_train = pd.concat([y_train, y_train_add], axis=0)

            test = X_test[(iter)*batch_size:batch_size*(1+iter)]

            xgbtunedreg.fit(X_train, y_train, verbose=True)
            preds = xgbtunedreg.predict(test)
            pred_y = pd.concat([pred_y, pd.Series(preds)],axis=0)

            best_params.append(xgbtunedreg.best_params_)

            joblib.dump(xgbtunedreg, 'result/task1_xgb_model_'+ str(iter) +'.pkl')
            
            print('now step =', iter)

    pred_y.reset_index(inplace=True)
    pred_y.drop('index',axis=1,inplace=True)

    mae = mean_absolute_error(y_test, pred_y)

    return pred_y, best_params, mae

pred_y, best_params, mae = make_result(train_df_final, test_df_final, param_grid, n_iter=800, batch_size=800)
    
result = []
for iter in range(len(X_test)-336) :

    mid = pred_y[0+iter:336+iter]
    result.append(mid)

result = np.array(result)
result = result.squeeze()
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

result.to_csv('task1_many_test.csv')