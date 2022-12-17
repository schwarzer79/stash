# -*- coding: utf-8 -*-
"""
task3_prepro_sw.py

@author: Sewon
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from pytimekr import pytimekr 

path = 'C:/ITWILL/Final_Project/datasets/data_4dist'
e = pd.read_csv(r'/data_tr_4dist.csv')
f = pd.read_csv(path + '/data_ts_4dist.csv')

train = e.copy() 
test = f.copy()

train.head() # 0  2017-01-01 01:00:00               525.0
train.tail() # 4  2017-01-01 05:00:00               330.0

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

train.isnull().sum() # 12
test.isnull().sum() # 337

# =============================================================================
# population 변수 추가 
# =============================================================================
gumi = pd.read_csv(path + '/gumi_pop.csv')
gumi.head()
gumi.tail()

# 2017
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 1), 'population'] = gumi.loc[0, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 2), 'population'] = gumi.loc[1, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 3), 'population'] = gumi.loc[2, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 4), 'population'] = gumi.loc[3, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 5), 'population'] = gumi.loc[4, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 6), 'population'] = gumi.loc[5, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 7), 'population'] = gumi.loc[6, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 8), 'population'] = gumi.loc[7, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 9), 'population'] = gumi.loc[8, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 10), 'population'] = gumi.loc[9, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 11), 'population'] = gumi.loc[10, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2017) & (train['ds'].dt.month == 12), 'population'] = gumi.loc[11, '산업군 인구수']

# 2018
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 1), 'population'] = gumi.loc[12, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 2), 'population'] = gumi.loc[13, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 3), 'population'] = gumi.loc[14, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 4), 'population'] = gumi.loc[15, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 5), 'population'] = gumi.loc[16, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 6), 'population'] = gumi.loc[17, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 7), 'population'] = gumi.loc[18, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 8), 'population'] = gumi.loc[19, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 9), 'population'] = gumi.loc[20, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 10), 'population'] = gumi.loc[21, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 11), 'population'] = gumi.loc[22, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2018) & (train['ds'].dt.month == 12), 'population'] = gumi.loc[23, '산업군 인구수']

# 2019
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 1), 'population'] = gumi.loc[24, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 2), 'population'] = gumi.loc[25, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 3), 'population'] = gumi.loc[26, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 4), 'population'] = gumi.loc[27, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 5), 'population'] = gumi.loc[28, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 6), 'population'] = gumi.loc[29, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 7), 'population'] = gumi.loc[30, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 8), 'population'] = gumi.loc[31, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 9), 'population'] = gumi.loc[32, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 10), 'population'] = gumi.loc[33, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 11), 'population'] = gumi.loc[34, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2019) & (train['ds'].dt.month == 12), 'population'] = gumi.loc[35, '산업군 인구수']

# 2020
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 1), 'population'] = gumi.loc[36, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 2), 'population'] = gumi.loc[37, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 3), 'population'] = gumi.loc[38, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 4), 'population'] = gumi.loc[39, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 5), 'population'] = gumi.loc[40, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 6), 'population'] = gumi.loc[41, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 7), 'population'] = gumi.loc[42, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 8), 'population'] = gumi.loc[43, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 9), 'population'] = gumi.loc[44, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 10), 'population'] = gumi.loc[45, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 11), 'population'] = gumi.loc[46, '산업군 인구수']
train.loc[(train['ds'].dt.year == 2020) & (train['ds'].dt.month == 12), 'population'] = gumi.loc[47, '산업군 인구수']


# 2021(test)
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 1), 'population'] = gumi.loc[48, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 2), 'population'] = gumi.loc[49, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 3), 'population'] = gumi.loc[50, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 4), 'population'] = gumi.loc[51, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 5), 'population'] = gumi.loc[52, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 6), 'population'] = gumi.loc[53, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 7), 'population'] = gumi.loc[54, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 8), 'population'] = gumi.loc[55, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 9), 'population'] = gumi.loc[56, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 10), 'population'] = gumi.loc[57, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 11), 'population'] = gumi.loc[58, '산업군 인구수']
test.loc[(test['ds'].dt.year == 2021) & (test['ds'].dt.month == 12), 'population'] = gumi.loc[59, '산업군 인구수']

train['population'].shape # (35063,)
test['population'].shape # (8761,)
test.isnull().sum()

# 결측치 처리
# 2022년 1월 1일 00:00:00 결측치를 직전 데이터 값으로 대체
test.loc[8760, 'population'] = test.loc[8759, 'population']
test # 49299.0으로 결측치 채워짐


# =============================================================================
# 기후 변수 추가
# =============================================================================
### train
train_gm = pd.read_csv(r'C:\Users\현이\Desktop\project\task3 기후데이터\train_gm_weather.csv')
train_gm
train_gm = train_gm.rename(columns = {'기온(°C)' : 'temp', '강수량(mm)' : 'rain', '습도(%)' : 'hum'})
train_gm['ds'] = pd.to_datetime(train_gm['ds'])

# train에 기후 변수 추가(병합)
train = pd.merge(train, train_gm, how = 'outer', on = 'ds')
train.info()

### test
test_gm = pd.read_csv(r'C:\Users\현이\Desktop\project\task3 기후데이터\test_gm_weather.csv')
test_gm
test_gm = test_gm.rename(columns = {'기온(°C)' : 'temp', '강수량(mm)' : 'rain', '습도(%)' : 'hum'})
test_gm['ds'] = pd.to_datetime(test_gm['ds'])

# test에 기후 변수 추가(병합)
test = pd.merge(test, test_gm, how = 'outer', on = 'ds')
test.info()


### 기후 데이터 이상치 확인 : 방재기상관측(AWS) 기준
'''
구미 기상기후 데이터 : 종관기상관측(AOS)
- ASOS 상한/하한 기준
기온 : [-80, 60]  
일강수량 : [0, 1000]
강수량 : [0, 300]
습도 : [0, 100]
'''

train[(train['temp'] < -80) | (train['temp'] > 60)] # 이상치 없음
train['temp'].describe()

test[(test['temp'] < -80) | (test['temp'] > 60)] # 이상치 없음
test['temp'].describe()

train[(train['rain'] < 0) | (train['rain'] > 300)] # 이상치 없음
train['rain'].describe()

test[(test['rain'] < 0) | (test['rain'] > 300)] # 이상치 없음
test['rain'].describe()

train[(train['hum'] < 0) | (train['hum'] > 100)] # 이상치 없음
train['hum'].describe()
# max        100.000000   <-- 상한값

test[(test['hum'] < 0) | (test['hum'] > 100)] # 이상치 없음
test['hum'].describe()

### 결측치 처리 (보간법으로 처리)
train.isnull().sum() 

test.isnull().sum()

# rain 결측치는 0으로
train['rain'] = train['rain'].fillna(0)
# 기온은 시간의 영향을 많이 받을 것으로 예상되므로 time 옵션을 지정
train['hum'] = train['hum'].interpolate(option='linear')
train['temp'] = train['temp'].interpolate(option='time')
train.isna().sum()

test['rain'] = test['rain'].fillna(0)
test['hum'] = test['hum'].interpolate(option='linear')
test['temp'] = test['temp'].interpolate(option='time')# 기온 : time 옵션
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
