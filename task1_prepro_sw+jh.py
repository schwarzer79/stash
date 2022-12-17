# -*- coding: utf-8 -*-
"""
task1_prepro_sw.py

@author: Sewon
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from pytimekr import pytimekr # 휴일 library

path = 'C:/ITWILL/Final_Project/datasets/data_city'
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
# week_weekend 변수 추가
# =============================================================================
train.ds.dt.weekday.unique() 
'''
6 : 일, 0 : 월, 1 : 화, 2 : 수, 3 : 목, 4 : 금, 5 : 토
'''

train['week'] = 'weekday' # weekday로 초기화
train.loc[(train.ds.dt.weekday == 5) | (train.ds.dt.weekday == 6), 'week'] = 'weekend'
train = pd.get_dummies(train, 'week')
train = train.drop('week_weekday', axis = 1)
train

test['week'] = 'weekday' # weekday로 초기화
test.loc[(test.ds.dt.weekday == 5) | (test.ds.dt.weekday == 6), 'week'] = 'weekend'
test = pd.get_dummies(test, 'week')
test = test.drop('week_weekday', axis = 1)
test

train.isnull().sum()
test.isnull().sum()

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
train_gc = train_gc.rename(columns = {'date' : 'ds', '기온(°C)' : 'temp', 
                                      '강수량(mm)' : 'rain', '습도(%)' : 'hum'})
train_gc['ds'] = pd.to_datetime(train_gc['ds'])

# train에 기후 변수 추가(병합)
train = pd.merge(train, train_gc, how = 'outer', on = 'ds')
train.info()

### test
test_gc = pd.read_csv(path + '/test_gc_weather.csv')
test_gc
test_gc = test_gc.drop('풍속(m/s)', axis = 1)
test_gc = test_gc.rename(columns = {'date' : 'ds', '기온(°C)' : 'temp', 
                                      '강수량(mm)' : 'rain', '습도(%)' : 'hum'})
test_gc = test_gc.rename(columns = {'date' : 'ds'})
test_gc['ds'] = pd.to_datetime(test_gc['ds'])

# test에 기후 변수 추가(병합)
test = pd.merge(test, test_gc, how = 'outer', on = 'ds')
test.info()

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
path2 = 'C:/ITWILL/Final_Project/datasets/data_4dist/sensory_temperature'

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
df = df.drop(['기온(°C)', '풍속(km/h)', '습도(%rh)'], axis = 1)
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
df2 = df2.drop(['기온(°C)', '풍속(km/h)', '습도(%rh)'], axis = 1)
df2.columns = ['date', 'st']
df2['date'] = pd.to_datetime(df2['date'])

# 결측치 채우기(보간법)
df2.isnull().sum() # 결측치 28개
df2['st'] = df2['st'].interpolate(option = 'linear') 

test = add_st(test, df2)
test.isnull().sum()      

train.to_csv(path + 'task1_prepro_sw.csv')
train.to_csv(path + 'task1_prepro_sw.csv')        


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
    data_holi['holiday'] = data_holi['holiday'].astype('int64')

    return(data_holi)
 
# holiday 변수 추가 
train = plus_holiday(train,holiday_df)
test = plus_holiday(test, holiday_df)