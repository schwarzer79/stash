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
from bs4 import BeautifulSoup
from lxml import html
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import scipy.stats
import matplotlib.dates as mdates
import copy
plt.style.use('bmh')
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#####################
## data import
#####################
train_df = pd.read_csv('dataset/data_city/task1_city_tr_data.csv')
test_df = pd.read_csv('dataset/data_city/task1_ts_final_data.csv')
sample_df = pd.read_csv('dataset/data_city/sample_city.csv')

def season_calc(month):
    if month in [5,6,7,8]:
        return "summer"
    else:
        return "winter"

def is_season(ds) :
    date = pd.to_datetime(ds)
    return (date.month > 10 or date.month < 4)

def feature_preprocessing(train_df) :
    # 불필요한 column 삭제
    train_df = train_df.drop(['Unnamed: 0', 'week_weekend', 'kovo_Away'], axis=1)

    # 'ds' 데이터형 변경
    train_df.ds = pd.to_datetime(train_df['ds'])

    # ds 기반 새로운 feature 생성
    train_df['month'] = train_df.ds.dt.month
    train_df['year'] = train_df.ds.dt.year
    train_df['day'] = train_df.ds.dt.day
    train_df['hour'] = train_df.ds.dt.hour
    train_df['dayofweek'] = train_df.ds.dt.dayofweek
    weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
    train_df['weekday'] = train_df.ds.dt.weekday.map(weekdays)
    train_df['match_season'] = train_df['ds'].apply(is_season)

    # 계절에 관한 feature 생성
    for iter in range(len(train_df)) :
        train_df.loc[iter,'season'] = season_calc(train_df.loc[iter, 'month'])

    # dtype 변경
    train_df['season'] = train_df['season'].astype('category')
    train_df['weekday'] = train_df['weekday'].astype('category')

    # 경기 유무
    for iter in train_df.index :
        if train_df.loc[iter, 'kovo_Home'] == True :
            train_df.loc[iter,'Kovo_match'] = 'on_match'
        else :
            train_df.loc[iter,'Kovo_match'] = 'off_match'

    train_df.drop(['kovo_Home','kovo_No'], axis=1, inplace=True)

    return train_df

train_df = feature_preprocessing(train_df)
test_df = feature_preprocessing(test_df)

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

train_df['y'].plot()
plt.show()

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

###########################
## EDA
###########################

## 각 데이터에 관한 EDA
# hour
train_df_Interpolation_time.groupby('hour')['y'].mean().plot(figsize = (10,5))
_ = plt.ylabel('Water Consumption')
_ = plt.ylim([0, max(train_df_Interpolation_time.groupby('hour')['y'].mean()) + 100])
_ = plt.xticks(train_df_Interpolation_time['hour'].unique())
_ = plt.title('Hourly Water Consumption averaged over 4 years (2017-20)')
_ = plt.show()

"""
03 ~ 06시가 가장 낮은 값을 가지며 출근시간 + 퇴근시간에 가장 많은 사용량을 보여줌
"""

# month
train_df_Interpolation_time.groupby('month')['y'].mean().plot(figsize = (10,5))
_ = plt.ylabel('Water Consumption')
_ = plt.ylim([0, max(train_df_Interpolation_time.groupby('month')['y'].mean()) + 100])
_ = plt.xticks(train_df_Interpolation_time['month'].unique())
_ = plt.title('Monthly Water Consumption averaged over 4 years (2017-20)')
_ = plt.show()

"""
5,6,7,8 월에 유의미한 상승치를 보여주고 있음 -> 이 기간을 summer season으로 지정해서 사용
"""

# 월 평균 소비량 plotting
train_dfc = train_df_Interpolation_time.groupby(['month','year'])['y'].mean()
train_dfc = pd.DataFrame(train_dfc)

#Unstacking
train_dfc_un = train_dfc.unstack(level = 'month')
train_dfc_un.head()
"""
                y
month          1           2           3           4           5           6           7           8           9           10          11          12
year
2017   174.481830  183.904762  191.377811  203.879016  313.475134  249.526389  232.466398  224.658602  214.697222  203.924731  212.947222  207.971774
2018   209.306452  212.125902  218.859543  222.566667  225.629810  251.565972  278.161972  289.489247  230.873611  229.677419  235.048431  229.646505
2019   229.674731  228.052955  237.954301  243.634722  253.514785  256.399306  266.525538  265.498656  248.229167  245.388441  248.808333  253.352151
2020   231.786880  244.392241  241.721774  245.951389  253.362903  278.147222  261.981183  269.813172  262.761111  256.790323  256.687500  251.866935
"""

train_dfc_un.reset_index(inplace = True)

melt = pd.melt(train_dfc_un, id_vars='year', value_name='Avg. monthly Water consumption')
melt.head()

# plotting
g = sns.FacetGrid(melt, col="month", margin_titles = True, col_wrap = 4)
g.map(plt.scatter, "year", "Avg. monthly Water consumption");
g.set_axis_labels("Years", "Avg. monthly MWH ");
g.set(ylim=(melt['Avg. monthly Water consumption'].min() - 200, melt['Avg. monthly Water consumption'].max() + 200));
_ = plt.subplots_adjust(top=0.9)
_ = g.fig.suptitle('Average monthly water consumption variation over years 2017 to 2020', fontsize = 16 )
_ = plt.show()

"""
전체적으로 연도가 지나갈수록 상승하는 추세를 가지고 있음
월간 차이는 조금씩 있지만 그렇게 큰 차이를 가지고 있지는 않음
"""

# plotting hourly vs weekdays water consumption
hour_weekday = train_df_Interpolation_time.pivot_table(values='y', index='hour', columns = 'weekday', aggfunc = 'mean')

#plotting a heatmap with a colorbar; the colorbar shows the energy consumption in MWH
_ = plt.figure(figsize=(12, 8))
ax = sns.heatmap(hour_weekday.sort_index(ascending = False), cmap='viridis')

#_ = plt.title('Average energy consumption in MWH for each hour of each weekday over the entire period')
_ = ax.set_title("Average water consumption in MWH for each hour of each weekday averaged over 4 years", fontsize = 14)
_ = plt.show()

"""
주말과 평일 간 출근시간 물 사용량이 큰 차이를 보여줌, 18 ~ 23시 사이에도 어느정도의 차이가 존재
"""

# water consumption의 전체적인 분포를 보기위한 히스토그램
_ = plt.figure(figsize = (12,8))
_ = sns.distplot(train_df_Interpolation_time['y'], kde=False)
_ = plt.title('Water consumption (MWH) distribution over 5 years (2017-20) for y utility region')
_ = plt.xlabel('Water consumption in MWH')
_ = plt.ylabel('count')
_ = plt.show()

"""
대부분 500이하의 값에 머물러 있는 경우가 많음
"""

# 연도의 물 사용량 분포 시각화
for year in train_df_Interpolation_time['year'].unique():
    train_df_Interpolation_time[train_df_Interpolation_time['year'] == year]['y'].plot(kind='density', figsize = (12,7), legend = True, label = year)
_ = plt.xlabel('Water consumption in MWH')
_ = plt.title('Variation in Distribution of Water consumption over 4 years (2017-20)')
_ = plt.show()

"""
작은 값에 대해서는 2017 -> 2020으로 갈수록 감소하고 큰 값에 대해서는 반대로 2020 -> 2017로 갈수록 감소한다. = 점차 증가하는 추세를 가지지만 그 폭이 그렇게 크지 않을 것
"""

# season별 연도 물 사용량 분포 시각화
season = ['summer', 'winter']
#color_names = ["tab:blue", "tab:orange", "tab:green", "tab:red" , "tab:purple"]
for i, season in enumerate(season):
    
    ax = plt.subplot(121+i)
    for j, year in enumerate(train_df_Interpolation_time['year'].unique()):
        _ = train_df_Interpolation_time[(train_df_Interpolation_time['year'] == year) & (train_df_Interpolation_time['season'] == season)]['y'].\
                                plot(kind='density', figsize=(15,7), legend=True, label= year, sharey=True)#, \
                                #c = color_names[j])
    _ = plt.title('Water consumption distribution across the years for '+season)
    _ = plt.ylim(-0.0001, 0.0050)
    _ = plt.xlim(-100, 6500)
    _ = plt.xlabel('Water consumption in MWH')

_ = plt.tight_layout()
_ = plt.show()

# 월별 최대 사용량 plotting
# Resampling the energy data monthly and calculating the max energy consumption for each month
monthly_en = train_df_Interpolation_time.resample('M', label = 'left')['y'].max()
_ = plt.figure(figsize = (15,6))
#plotting the max monthly energy consumption
_ = plt.plot(monthly_en)
# ensuring the limits on x axis to be between the dataframe's datetime limits
_ = plt.xlim(monthly_en.index.min(), monthly_en.index.max())
# Using matplotlib MonthLocator to be used in the xticks to mark individual months
locator = mdates.MonthLocator(bymonthday = 1, interval = 2)  # every 2 months 
fmt = mdates.DateFormatter('%m-%y')  # xticks to be displayed as 01-14 (i.e. Jan'14) and so on
X = plt.gca().xaxis
# Setting the locator
X.set_major_locator(locator)
# Specify formatter
X.set_major_formatter(fmt)
_ = plt.xticks(rotation = 60)
_ = plt.ylabel('Max Water consumption in MWH')
_ = plt.xlabel('Date')
#_ = plt.figure(figsize=(15,6))
#_ = SDGE_t_pv['SDGE'].rolling(24*30).max().plot()
_ = plt.show()

# 매 시간마다의 연도별 물 사용량 plotting
f = plt.figure(figsize = (20,40))
peak_hours = np.arange(0,24) 
for i, hour in enumerate(peak_hours):
    ax = f.add_subplot(12,2,i+1)
    for j, year in enumerate(train_df_Interpolation_time['year'].unique()):
        train_df_Interpolation_time[(train_df_Interpolation_time['year'] == year) & (train_df_Interpolation_time['hour'] == hour)]['y'].\
                                plot(kind='density',  sharey=False, legend=True, label= year)# ,\
                                #c = color_names[j])
    plt.title('Energy consumption distribution across the years for hour '+str(hour))
    plt.xlim(np.min(train_df_Interpolation_time.y.values), np.max(train_df_Interpolation_time.y.values))
plt.tight_layout()
plt.show()

# 날씨 데이터와 물 사용량 간의 plotting
# Plotting the energy and weather data on the same graph as line plots
fig, ax1 = plt.subplots(figsize=(15,7))
rolling_num = 24*30 # smoothing the data a bit by taking the mean of last 'rolling_num' values 
#i.e. plotting the 30 day average energy consumption and temperature values 
color = 'tab:red'
ax1.set_xlabel('Dates')
ax1.set_ylabel('Water consumption MWH', color = color)
ax1.plot(train_df_Interpolation_time['y'].rolling(rolling_num).mean(), color = color, alpha = 0.5)        
ax1.tick_params(axis='y', labelcolor = color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Dry Bulb Temp F', color = color)  # we already handled the x-label with ax1
ax2.plot(train_df_Interpolation_time['temp'].rolling(rolling_num).mean(), color = color, alpha = 0.5)   
ax2.tick_params(axis='y', labelcolor = color)

fig.suptitle('Water consumption and temperature plotted together', fontsize = 14)
fig.tight_layout()
plt.show()

"""
완벽하게 일치하지는 않지만 어느정도의 추세는 일치하는 모습
"""

# 상관계수
scipy.stats.pearsonr(train_df_Interpolation_time['y'], train_df_Interpolation_time['temp']) # (0.21699297504740508, 0.0)

for season in train_df_Interpolation_time['season'].unique():
    corrcoef, pvalue = scipy.stats.pearsonr(train_df_Interpolation_time[train_df_Interpolation_time['season'] == season]['y'], \
                                            train_df_Interpolation_time[train_df_Interpolation_time['season'] == season]['temp'])
    print('pearson correlation coefficient and pvalue for '+season, corrcoef, pvalue)

"""
pearson correlation coefficient and pvalue for winter 0.19760269382662266 2.0411627840616805e-203
pearson correlation coefficient and pvalue for summer 0.268723511639899 1.9283242052351726e-194
"""

# 인구와 물 사용량 간의 상관계수

scipy.stats.pearsonr(train_df_Interpolation_time['y'], train_df_Interpolation_time['population']) # (0.09460661355515407, 1.5936758659889704e-70)

# 인구 plotting
_ = train_df_Interpolation_time.population.plot(figsize=(10,5))
_ = plt.legend()
_ = plt.title('Population in y territory from 2017 to 2020')
_ = plt.ylabel('Population')
_ = plt.show()


#################################
## Analysis
#################################

# pred vs true 값 비교 plotting function
def plot_predvstrue_reg(pred, truth, model_name=None):
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.scatter(truth, pred) 
    _ = plt.xlabel("Observed Water Consumption")
    _ = plt.ylabel("Predicted Water consumption")
    _ = plt.title("Observed vs Predicted water consumption using model {}".format(model_name))
    _ = plt.xlim(1000, 5000)
    _ = plt.ylim(1000, 5000)
    #plotting 45 deg line to see how the prediction differs from the observed values
    x = np.linspace(*ax.get_xlim())
    _ = ax.plot(x, x)
    _ = fig.show()


# RMSE, MAE, R2score, MAPE
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

# 주어진 시계열 plotting
def plot_timeseries(ts, title = 'og', opacity = 1):
    """
    Plot plotly time series of any given timeseries ts
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = ts.index, y = ts.values, name = "observed",
                         line_color = 'lightslategrey', opacity = opacity))

    fig.update_layout(title_text = title,
                  xaxis_rangeslider_visible = True)
    fig.show()

# 예측 시계열 plotting
def plot_ts_pred(og_ts, pred_ts, model_name=None, og_ts_opacity = 0.5, pred_ts_opacity = 0.5):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = og_ts.index, y = np.array(og_ts.values), name = "Observed",
                         line_color = 'deepskyblue', opacity = og_ts_opacity))

    try:
        fig.add_trace(go.Scatter(x = pred_ts.index, y = pred_ts, name = model_name,
                         line_color = 'lightslategrey', opacity = pred_ts_opacity))
    except: #if predicted values are a numpy array they won't have an index
        fig.add_trace(go.Scatter(x = og_ts.index, y = pred_ts, name = model_name,
                         line_color = 'lightslategrey', opacity = pred_ts_opacity))


    #fig.add_trace(go)
    fig.update_layout(title_text = 'Observed test set vs predicted energy MWH values using {}'.format(model_name),
                  xaxis_rangeslider_visible = True)
    fig.show()

# 
def train_test(data, test_size = 0.15, scale = False, cols_to_transform=None, include_test_scale=False):
    df = data.copy()
    # get the index after which test set starts
    test_index = int(len(df)*(1-test_size))
    
    # StandardScaler fit on the entire dataset
    if scale and include_test_scale:
        scaler = StandardScaler()
        df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])
        
    X_train = df.drop('y', axis = 1).iloc[:test_index]
    y_train = df.y.iloc[:test_index]
    X_test = df.drop('y', axis = 1).iloc[test_index:]
    y_test = df.y.iloc[test_index:]
    
    # StandardScaler fit only on the training set
    if scale and not include_test_scale:
        scaler = StandardScaler()
        X_train[cols_to_transform] = scaler.fit_transform(X_train[cols_to_transform])
        X_test[cols_to_transform] = scaler.transform(X_test[cols_to_transform])
    
    return X_train, X_test, y_train, y_test

##################################################################################################
# Simple Linear Regression

# creating categorical columns for linear regression 
train_df_Interpolation_time.drop('dayofweek', axis=1, inplace=True)
cat_cols = ['year', 'month', 'day', 'hour', 'weekday', 'season', 'Kovo_match', 'match_season', 'week_weekday']

for col in cat_cols:
    train_df_Interpolation_time[col] = train_df_Interpolation_time[col].astype('category')

train_df_final = train_df_Interpolation_time

# dummy
train_df_final_lin = pd.get_dummies(train_df_final, drop_first=True)

train_df_final_lin.columns

# linear regression model fitting

m = ols('y ~  C(year) + C(month) + C(hour) + C(season) + C(weekday) + C(Kovo_match)\
                 + population + temp + rain', train_df_final).fit()
print(m.summary())

"""
                           OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.402
Model:                            OLS   Adj. R-squared:                  0.401
Method:                 Least Squares   F-statistic:                     500.8
Date:                Thu, 01 Dec 2022   Prob (F-statistic):               0.00
Time:                        16:36:38   Log-Likelihood:            -2.2002e+05
No. Observations:               35063   AIC:                         4.401e+05
Df Residuals:                   35015   BIC:                         4.405e+05
Df Model:                          47
Covariance Type:            nonrobust
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                    99.0584     42.099      2.353      0.019      16.543     181.573
C(year)[T.2018]               3.9087      6.162      0.634      0.526      -8.169      15.986
C(year)[T.2019]               8.9329      8.871      1.007      0.314      -8.454      26.320
C(year)[T.2020]              10.7062     10.565      1.013      0.311     -10.002      31.415
C(month)[T.2]                 2.9538      3.499      0.844      0.399      -3.904       9.812
C(month)[T.3]                 0.5042      3.779      0.133      0.894      -6.903       7.912
C(month)[T.4]                 0.3543      4.339      0.082      0.935      -8.149       8.858
C(month)[T.5]                29.8339      8.741      3.413      0.001      12.702      46.966
C(month)[T.6]                22.9437      9.047      2.536      0.011       5.211      40.676
C(month)[T.7]                20.3312      9.340      2.177      0.030       2.025      38.638
C(month)[T.8]                21.2363      9.683      2.193      0.028       2.256      40.216
C(month)[T.9]                -0.6278      5.721     -0.110      0.913     -11.841      10.585
C(month)[T.10]               -0.0683      5.197     -0.013      0.990     -10.255      10.118
C(month)[T.11]               11.2389      4.986      2.254      0.024       1.466      21.012
C(month)[T.12]               15.4316      4.976      3.101      0.002       5.679      25.184
C(hour)[T.1]                -81.7357      4.760    -17.171      0.000     -91.065     -72.406
C(hour)[T.2]               -147.0726      4.762    -30.886      0.000    -156.406    -137.739
C(hour)[T.3]               -188.8628      4.764    -39.641      0.000    -198.201    -179.525
C(hour)[T.4]               -208.2507      4.768    -43.681      0.000    -217.595    -198.906
C(hour)[T.5]               -215.1968      4.771    -45.109      0.000    -224.547    -205.846
C(hour)[T.6]               -207.2444      4.773    -43.424      0.000    -216.599    -197.890
C(hour)[T.7]               -167.3334      4.768    -35.097      0.000    -176.678    -157.989
C(hour)[T.8]                -30.6024      4.760     -6.430      0.000     -39.931     -21.273
C(hour)[T.9]                 58.2900      4.771     12.218      0.000      48.939      67.641
C(hour)[T.10]                36.7930      4.817      7.639      0.000      27.352      46.234
C(hour)[T.11]                37.7272      4.876      7.738      0.000      28.171      47.283
C(hour)[T.12]                35.7230      4.929      7.247      0.000      26.061      45.385
C(hour)[T.13]                32.8153      4.972      6.600      0.000      23.070      42.561
C(hour)[T.14]                30.0135      4.999      6.004      0.000      20.215      39.812
C(hour)[T.15]                -4.3128      5.010     -0.861      0.389     -14.132       5.506
C(hour)[T.16]               -34.2230      4.999     -6.846      0.000     -44.021     -24.425
C(hour)[T.17]               -34.7190      4.957     -7.004      0.000     -44.435     -25.003
C(hour)[T.18]               -23.3754      4.889     -4.781      0.000     -32.958     -13.792
C(hour)[T.19]                15.9362      4.827      3.301      0.001       6.474      25.398
C(hour)[T.20]                65.6163      4.790     13.700      0.000      56.228      75.004
C(hour)[T.21]                83.0045      4.773     17.392      0.000      73.650      92.359
C(hour)[T.22]                94.3663      4.764     19.807      0.000      85.028     103.705
C(hour)[T.23]                75.1347      4.760     15.783      0.000      65.804      84.465
C(season)[T.winter]           4.7134      7.520      0.627      0.531     -10.027      19.454
C(weekday)[T.Monday]          4.6437      2.573      1.805      0.071      -0.399       9.686
C(weekday)[T.Saturday]      -22.0839      2.575     -8.576      0.000     -27.131     -17.037
C(weekday)[T.Sunday]        -10.7952      2.576     -4.191      0.000     -15.844      -5.746
C(weekday)[T.Thursday]        7.5303      2.572      2.928      0.003       2.490      12.571
C(weekday)[T.Tuesday]         9.8766      2.572      3.840      0.000       4.835      14.918
C(weekday)[T.Wednesday]       8.4725      2.580      3.284      0.001       3.416      13.528
C(Kovo_match)[T.on_match]     0.4666      3.692      0.126      0.899      -6.769       7.702
population                    0.0068      0.003      2.474      0.013       0.001       0.012
temp                          1.0707      0.168      6.357      0.000       0.741       1.401
rain                         -2.8884      0.669     -4.317      0.000      -4.200      -1.577
==============================================================================
Omnibus:                    15027.116   Durbin-Watson:                   1.190
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           263330.184
Skew:                           1.615   Prob(JB):                         0.00
Kurtosis:                      16.031   Cond. No.                     1.30e+16
==============================================================================
"""

# 필요한 컬럼만 남기고 다 삭제
train_df_final_lin.drop(['day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7',
       'day_8', 'day_9', 'day_10', 'day_11', 'day_12', 'day_13', 'day_14',
       'day_15', 'day_16', 'day_17', 'day_18', 'day_19', 'day_20', 'day_21',
       'day_22', 'day_23', 'day_24', 'day_25', 'day_26', 'day_27', 'day_28',
       'day_29', 'day_30', 'day_31', 'season_winter'], axis=1, inplace=True)

train_df_final_lin.columns

cols_to_transform = ['population', 'temp', 'rain','hum'] # other columns are binary values
X_train, X_test, y_train, y_test = train_test(train_df_final_lin, test_size = 0.15, scale = True, cols_to_transform=cols_to_transform)

# This creates a LinearRegression object
lm = LinearRegression()
lm

# Fitting the linear regression model
lm.fit(X_train, y_train)

# Plotting the coefficients to check the importance of each coefficient 

# Plot the coefficients
_ = plt.figure(figsize = (16, 7))
_ = plt.plot(range(len(X_train.columns)), lm.coef_)
_ = plt.xticks(range(len(X_train.columns)), X_train.columns.values, rotation = 90)
_ = plt.margins(0.02)
_ = plt.axhline(0, linewidth = 0.5, color = 'r')
_ = plt.title('sklearn Simple linear regression coefficients')
_ = plt.ylabel('lm_coeff')
_ = plt.xlabel('Features')
_ = plt.show()

plot_predvstrue_reg(lm.predict(X_test), y_test, model_name = 'sklearn Simple linear regression')

error_metrics(lm.predict(X_train), y_train, model_name = 'Simple linear regression with scaling', test = False)
"""
Error metrics for model Simple linear regression with scaling
RMSE or Root mean squared error: 129.88
Variance score: 0.39
Mean Absolute Error: 95.86
Mean Absolute Percentage Error: 77.28 %
"""

# on test set
error_metrics(lm.predict(X_test), y_test, model_name = 'Simple linear regression with scaling', test = True)
"""
Error metrics for model Simple linear regression with scaling
RMSE or Root mean squared error: 121.45
Variance score: 0.47
Mean Absolute Error: 94.05
Mean Absolute Percentage Error: 66.73 %
"""

# Plotting the predicted values with the original time series (test set)
plot_ts_pred(y_test, lm.predict(X_test), model_name='Simple linear regression with scaling', 
             og_ts_opacity = 0.5, pred_ts_opacity = 0.5)

##################################################################################
## 2. Ridge Regression

from sklearn.linear_model import Ridge

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha = 0.2, normalize = True) #selecting alpha=0.2 here. 
# Tried different values from 0.1 to 0.8, results don't change much

# Fit the regressor to the data
ridge.fit(X_train, y_train)

# Compute and print the coefficients
ridge_coef = ridge.coef_
#print(ridge_coef)

# Plot the coefficients
_ = plt.figure(figsize = (16, 7))
_ = plt.plot(range(len(X_train.columns)), ridge_coef)
_ = plt.xticks(range(len(X_train.columns)), X_train.columns.values, rotation = 90)
_ = plt.margins(0.02)
_ = plt.axhline(0, linewidth = 0.5, color = 'r')
_ = plt.title('significane of features as per Ridge regularization')
_ = plt.ylabel('Ridge coeff')
_ = plt.xlabel('Features')
_ = plt.show()

# PLotting the residuals
residuals = (y_test - ridge.predict(X_test))
_ = plt.figure(figsize=(7,7))
_ = plt.scatter(ridge.predict(X_test) , residuals, alpha = 0.5) 
_ = plt.xlabel("Model predicted energy values")
_ = plt.ylabel("Residuals")
_ = plt.title("Fitted values versus Residuals for Ridge regression")
_ = plt.show()

# error metrics
print('Ridge regression on training set')
error_metrics(ridge.predict(X_train), y_train, model_name = 'Ridge regression with scaling', test = False)

"""
RMSE or Root mean squared error: 131.03
Variance score: 0.38
Mean Absolute Error: 98.42
Mean Absolute Percentage Error: 91.12 %
"""

print('\nRidge regression on test set')
error_metrics(ridge.predict(X_test), y_test, model_name = 'Ridge regression with scaling', test = True)

"""
Error metrics for model Ridge regression with scaling
RMSE or Root mean squared error: 124.17
Variance score: 0.45
Mean Absolute Error: 97.45
Mean Absolute Percentage Error: 76.00 %
"""

# Plotting the observed test energy and predicted energy data on the same graph as line plots

plot_ts_pred(y_test, ridge.predict(X_test), model_name='Ridge regression', og_ts_opacity = 0.5, pred_ts_opacity = 0.5)

## feature 수 줄이기
train_df_1 = train_df_Interpolation_time

# hour 를 night, morning, afternoon, evening으로 분류
hour_dict = {'morning': list(np.arange(6,12)),'afternoon': list(np.arange(12,18)), 'evening': list(np.arange(18,24)),
            'night': [0, 1, 2, 3, 4, 5]}
hour_dict

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

train_df_1['time_of_day'] = train_df_1['hour'].apply(time_of_day)

cat_cols1 = ['month', 'day', 'hour', 'weekday', 'season', 'Kovo_match', 'match_season', 'week_weekday', 'time_of_day']
#not including year above to capture the decreasing energy trend over increasing value of years
for col in cat_cols1:
    train_df_1[col] = train_df_1[col].astype('category')

train_df_1['year'] = train_df_1['year'].astype(np.int64)

# Columns to use for regression
cols_use = ['y', 'year', 'month', 'time_of_day', 'season', 'week_weekday', 'Kovo_match', 'population', 'temp', 'rain', 'hum']

train_df_1_lin = pd.get_dummies(train_df_1[cols_use], drop_first = True)
train_df_1_lin.head()

cols_to_transform = ['population', 'temp', 'rain', 'hum', 'year']
X_train, X_test, y_train, y_test = train_test(train_df_1_lin, test_size = 0.15, scale = True, cols_to_transform=cols_to_transform, 
                                              include_test_scale=False)

#################################################################################
### 3. Elastic Net
# Trying elastic net regression

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def trend_model(data, cols_to_transform, l1_space, alpha_space, cols_use = 0, scale = True, test_size = 0.15, 
                include_test_scale=False):

    # Creating the train test split
    if cols_use != 0:
        df = data[cols_use]
    else:
        df = data
    
    X_train, X_test, y_train, y_test = train_test(df, test_size = test_size, 
                                              scale = scale, cols_to_transform=cols_to_transform, 
                                              include_test_scale=include_test_scale)

    
    # Create the hyperparameter grid
    #l1_space = np.linspace(0, 1, 50)
    param_grid = {'l1_ratio': l1_space, 'alpha': alpha_space}

    # Instantiate the ElasticNet regressor: elastic_net
    elastic_net = ElasticNet()

    # for time-series cross-validation set 5 folds
    tscv = TimeSeriesSplit(n_splits=5)

    # Setup the GridSearchCV object: gm_cv ...trying 5 fold cross validation 
    gm_cv = GridSearchCV(elastic_net, param_grid, cv = tscv)

    # Fit it to the training data
    gm_cv.fit(X_train, y_train)

    # Predict on the test set and compute metrics
    y_pred = gm_cv.predict(X_test)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
    print("Tuned ElasticNet R squared: {}".format(r2))
    print("Tuned ElasticNet RMSE: {}".format(np.sqrt(mse)))
    # fitting the elastic net again using the best model from above

    elastic_net_opt = ElasticNet(l1_ratio = gm_cv.best_params_['l1_ratio']) 
    elastic_net_opt.fit(X_train, y_train)
    
    # Plot the coefficients
    _ = plt.figure(figsize = (15, 7))
    _ = plt.plot(range(len(X_train.columns)), elastic_net_opt.coef_)
    _ = plt.xticks(range(len(X_train.columns)), X_train.columns.values, rotation = 90)
    _ = plt.margins(0.02)
    _ = plt.axhline(0, linewidth = 0.5, color = 'r')
    _ = plt.title('significane of features as per Elastic regularization')
    _ = plt.ylabel('Elastic net coeff')
    _ = plt.xlabel('Features')
    _ = plt.show()
    
    # Plotting y_true vs predicted
    _ = plt.figure(figsize = (5,5))
    _ = plot_predvstrue_reg(elastic_net_opt.predict(X_test), y_test, model_name = 'Elastic net optimal linear regression')
    _ = plt.show()
    
    # returns the train and test X and y sets and also the optimal model
    return X_train, X_test, y_train, y_test, elastic_net_opt

data = train_df_1_lin
cols_to_transform = ['population', 'temp', 'rain', 'hum', 'year']
l1_space = np.linspace(0, 1, 30)
alpha_space = np.logspace(-2, 0, 30)

# Fitting, tuning and predicting using the best elastic net regression model
import warnings  
warnings.filterwarnings('ignore')
X_train, X_test, y_train, y_test, elastic_net_opt = trend_model(data=data, cols_to_transform=cols_to_transform, 
                                                                l1_space=l1_space, alpha_space=alpha_space,
                                                                scale = True, test_size = 0.15, include_test_scale=False)

"""
Tuned ElasticNet l1 ratio: {'alpha': 0.4520353656360243, 'l1_ratio': 1.0}
Tuned ElasticNet R squared: 0.29475574946613337
Tuned ElasticNet RMSE: 140.00179453768655
"""

# Plotting the observed test energy and predicted energy data on the same graph as line plots
plot_ts_pred(y_test, elastic_net_opt.predict(X_test), model_name='Optimal Elastic net regression', \
             og_ts_opacity = 0.5, pred_ts_opacity = 0.5)

# printing the error metrics
print('Elastic net regression on training set')
error_metrics(elastic_net_opt.predict(X_train), y_train, model_name = 'Tuned elastic net regression with reduced hour space', 
              test = False)

"""
Error metrics for model Tuned elastic net regression with reduced hour space
RMSE or Root mean squared error: 146.85
Variance score: 0.22
Mean Absolute Error: 114.22
Mean Absolute Percentage Error: 130.49 %
"""

print('\nElastic net regression on test set')
error_metrics(elastic_net_opt.predict(X_test), y_test, model_name = 'Tuned elastic net regression with reduced hour space', 
              test = True)

"""
Error metrics for model Tuned elastic net regression with reduced hour space
RMSE or Root mean squared error: 140.48
Variance score: 0.29
Mean Absolute Error: 113.25
Mean Absolute Percentage Error: 106.19 %
"""

#################################################################################
## 4. RandomForest Regression
from sklearn.ensemble import RandomForestRegressor

# Tuning Random forest
# n_estimators = number of trees in the forest
# max_features = max number of features considered for splitting a node

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(10, 200, 10, endpoint=True)]
max_features = ['auto', 'sqrt']
max_depth = list(range(1,6))
# Create the random grid
random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth':max_depth}
print(random_grid)

#import randomsearchcv
from sklearn.model_selection import RandomizedSearchCV

# First create the base model to tune
rf = RandomForestRegressor()

# Creating a time series split as discussed in the Introduction
tscv = TimeSeriesSplit(n_splits=5)
# Random search of parameters
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               cv = tscv, verbose=2, random_state = 42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_ # {'n_estimators': 73, 'max_features': 'auto', 'max_depth': 5}

rf_random.score(X_train, y_train) # 0.2734860244471772
rf_random.score(X_test, y_test) # 0.34299576911292895

plot_ts_pred(y_test, rf_random.predict(X_test), model_name='Tuned Random forest with reduced hour space', \
             og_ts_opacity = 0.5, pred_ts_opacity = 0.5)

# Random forest error metrics

print('Tuned Random forest errors on training set')
error_metrics(rf_random.predict(X_train), y_train, model_name = 'Tuned Random forest with reduced hour space', test = False)
"""
Error metrics for model Tuned Random forest with reduced hour space
RMSE or Root mean squared error: 141.29
Variance score: 0.27
Mean Absolute Error: 109.81
Mean Absolute Percentage Error: 120.94 %
"""
print('\nTuned Random forest errors on test set')
error_metrics(rf_random.predict(X_test), y_test, model_name = 'Tuned Random forest with reduced hour space', test = True)
"""
Error metrics for model Tuned Random forest with reduced hour space
RMSE or Root mean squared error: 135.13
Variance score: 0.34
Mean Absolute Error: 106.24
Mean Absolute Percentage Error: 84.05 %
"""

#################################################################################
## 5. X 변수에 lag추가
train_df_1_lin.head(2)

# Adding max 24 lags; lag1 is the value of the energy consumption in the previous hour, lag2 is the value of energy consumption..
#..2 hours before the current value and so on.

# Creating the lag variables
for i in range(24):
    train_df_1_lin['lag'+str(i+1)] = train_df_1_lin['y'].shift(i+1)

lag_y = train_df_1_lin.dropna()
lag_y.head(2)

# lag data에 elsetic net 적합
cols_to_transform = ['population', 'temp', 'rain', 'hum', 'year']

# Adding the energy consumption lags to the columns to transform 
list_lags = ['lag'+str(i+1) for i in range(24)]
cols_to_transform.extend(list_lags)

X_train_lag, X_test_lag, y_train_lag, y_test_lag = train_test(lag_y, \
                                                              test_size = 0.15, scale = True, \
                                                              cols_to_transform=cols_to_transform)

elastic_net_lag = ElasticNet(l1_ratio = 1, alpha=0.2)
elastic_net_lag.fit(X_train_lag, y_train_lag)

# Plot the coefficients
_ = plt.figure(figsize = (16, 7))
_ = plt.plot(range(len(X_train_lag.columns)), elastic_net_lag.coef_)
_ = plt.xticks(range(len(X_train_lag.columns)), X_train_lag.columns.values, rotation = 90)
_ = plt.margins(0.02)
_ = plt.axhline(0, linewidth = 0.5, color = 'r')
_ = plt.title('significane of features as per Elastic regularization with scaling and including lags')
_ = plt.ylabel('Linear reg coeff')
_ = plt.xlabel('Features')
_ = plt.show()

X_train_lag.columns.values

# random forest에 lag 적합
rf = RandomForestRegressor()
tscv = TimeSeriesSplit(n_splits=5)
# Random search of parameters
rflag = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               cv = tscv, verbose=2, random_state = 42, n_jobs = -1)

# Fit the random search model
rflag.fit(X_train_lag, y_train_lag)

rflag.best_params_

# error metrics
print('Random forest errors with lags on training')
error_metrics(rflag.predict(X_train_lag), y_train_lag, model_name = 'Random forest with all lags', test=False)

"""
Error metrics for model Random forest with all lags
RMSE or Root mean squared error: 118.39
Variance score: 0.49
Mean Absolute Error: 91.63
Mean Absolute Percentage Error: 75.88 %
"""

print('\nRandom forest errors with lags on test')
error_metrics(rflag.predict(X_test_lag), y_test_lag, model_name = 'Random forest with all lags', test=True)
"""
Error metrics for model Random forest with all lags
RMSE or Root mean squared error: 118.16
Variance score: 0.50
Mean Absolute Error: 90.00
Mean Absolute Percentage Error: 55.70 %
"""

print('\nElastic net errors with lags on train')
error_metrics(elastic_net_lag.predict(X_train_lag), y_train_lag, model_name = 'Elastic net with all lags', test=False)

"""
Error metrics for model Elastic net with all lags
RMSE or Root mean squared error: 120.18
Variance score: 0.47
Mean Absolute Error: 92.18
Mean Absolute Percentage Error: 79.78 %
"""

print('\nElastic net errors with lags on test')
error_metrics(elastic_net_lag.predict(X_test_lag), y_test_lag, model_name = 'Elastic net with all lags', test=True)

"""
Error metrics for model Elastic net with all lags
RMSE or Root mean squared error: 114.09
Variance score: 0.53
Mean Absolute Error: 88.83
Mean Absolute Percentage Error: 62.79 %
"""

#################################################################################
### 6. Time Series Features and Models

# Creating a simple time series dataframe
ts_sdge = pd.DataFrame(train_df_1_lin["y"], columns=['y'])
ts_sdge.tail()

plot_timeseries(ts_sdge['y'], title = 'Original data set')

# decompose
from statsmodels.graphics import tsaplots
import statsmodels.api as sm

decomp = sm.tsa.seasonal_decompose(ts_sdge['y'])
print(decomp.seasonal.head()) # checking the seasonal component
_ = decomp.plot()
_ = plt.show()

ts_sdge['seasonal'] = decomp.seasonal
plot_timeseries(ts_sdge['seasonal'], title = 'Seasonal component')

ts_sdge['trend'] = decomp.trend
ts_sdge['trend'].dropna().plot()
plt.show()

_ = train_df_Interpolation_time['y'].rolling(window = 24*30*12).mean().plot(figsize=(12,5))
_ = plt.title('Checking trend in the data by averaging yearly values')
_ = plt.show()

# Plotting the quartely rolling MAX of the time series to check trend
_ = train_df_Interpolation_time['y'].rolling(window = 24*30*12).max().plot(figsize=(12,5))
_ = plt.title('Checking trend in the data by taking the MAX of yearly values')
_ = plt.show()

decomp_diff1 = sm.tsa.seasonal_decompose(ts_sdge['y'].diff().dropna()) 
_ = decomp_diff1.plot()
_ = plt.show()

decomp_diff24 = sm.tsa.seasonal_decompose(ts_sdge['y'].diff().dropna().diff(24).dropna()) 
_ = decomp_diff24.plot()
_ = plt.show()

# dicky fuller test
from statsmodels.tsa.stattools import adfuller

def run_adfuller(ts):
    result = adfuller(ts)
    # Print test statistic
    print("t-stat", result[0])
    # Print p-value
    print("p-value", result[1])
    # Print #lags used
    print("#lags used", result[2])
    # Print critical values
    print("critical values", result[4]) 

print("for no differencing\n")
run_adfuller(ts_sdge['y']) 
"""
t-stat -10.55075984249137
p-value 8.187833715543524e-19
#lags used 51
critical values {'1%': -3.43053679213716, '5%': -2.8616225575095284, '10%': -2.566813942767471}
"""
print("\nfor single differencing\n")
run_adfuller(ts_sdge['y'].diff().dropna())

"""
t-stat -53.833801719303175
p-value 0.0
#lags used 50
critical values {'1%': -3.43053679213716, '5%': -2.8616225575095284, '10%': -2.566813942767471}
"""

print("\nfor differenced data set over lags 24 after single differencing\n")
run_adfuller(ts_sdge['y'].diff().dropna().diff(24).dropna())

"""
t-stat -47.207206353501455
p-value 0.0
#lags used 52
critical values {'1%': -3.430536930966715, '5%': -2.861622618866829, '10%': -2.56681397542637}
"""


## 추세를 보기 위한 지수 평활화
def plot_ewma(ts, alpha):
    expw_ma = ts.ewm(alpha=alpha).mean()
    
    plot_ts_pred(ts, expw_ma, model_name='Exponentially smoothed data with alpha = {}'.format(alpha), \
                 og_ts_opacity = 0.5, pred_ts_opacity = 0.5)

plot_ewma(ts_sdge['y'], 0.3)

## ACF plotting

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def acf_pacf_plots(ts, lags, figsize = (12,8)):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = figsize)
    
    # Plot the ACF of ts
    _ = plot_acf(ts, lags = lags, zero = False, ax = ax1, alpha = 0.05)

    # Plot the PACF of ts
    _ = plot_pacf(ts, lags = lags, zero = False, ax = ax2, alpha = 0.05)
    _ = plt.show()

dfacf = []
dfacf = ts_sdge['y']
lags = 50

acf_pacf_plots(dfacf, lags = lags, figsize = (12,8))

dfacf = []
dfacf = ts_sdge['y']
dfacf = dfacf.diff().dropna() 
dfacf = dfacf.diff(24).dropna()
dfacf = dfacf.diff(24*365).dropna()
lags = 100

acf_pacf_plots(dfacf, lags = lags, figsize = (12,8)) # 차분을 했음에도 계절성이 남아있는 모습 = multiple seasonailty

#################################################################################
# multiple seasonality 다루기
# SARIMAX 를 위한 푸리에 주기 추가
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

add_fourier_terms(lag_y, year_k= 5, week_k=5 , day_k=5)

ycyc = lag_y.drop([col for col in lag_y if 
                         col.startswith('time') or col.startswith('season') or col.startswith('lag')], axis=1)

###########################################
# SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

ycyc.columns

cols_to_transform = ['population', 'temp', 'rain', 'hum']  
X_train_lag, X_test_lag, y_train_lag, y_test_lag = train_test(ycyc, 
                                                              test_size = 0.15, scale = True, 
                                                              cols_to_transform=cols_to_transform)

# Since ARIMA model uses the past lag y values, scaling the energy values as well. 
#i.e. fit the scaler on y_train and transform it and also transform y_test using the same scaler if required later

scaler1 = StandardScaler()
y_train_lag = pd.DataFrame(scaler1.fit_transform(y_train_lag.values.reshape(-1,1)), index = y_train_lag.index, 
                           columns = ['y'])
# y_test_lag = scaler1.transform(y_test_lag)

model_opt = SARIMAX(y_train_lag, order=(2,1,1), seasonal_order=(1, 0, 1, 24), exog = X_train_lag, trend='c')
results = model_opt.fit()

pred = results.get_prediction(start=X_train_lag.index[-24*7], end=X_train_lag.index[-1], 
                              dynamic=True, exog=X_train_lag.iloc[-24*7:, :])
pred_ci = pred.conf_int()

pred1 = scaler1.inverse_transform(pred.predicted_mean) # error here
pred_ci1 = scaler1.inverse_transform(pred.conf_int())

y_actual_train = np.squeeze(scaler1.inverse_transform(y_train_lag))
y_actual_train = pd.Series(y_actual_train, index = X_train_lag.index )

pred1 = pd.Series(pred1, index = X_train_lag.iloc[-24*7:, :].index )
pred_ci1 = pd.DataFrame(pred_ci1, index = pred1.index, columns = ['lower y', 'upper y'])

lower_limits = pred_ci1.loc[:,'lower y']
upper_limits = pred_ci1.loc[:,'upper y']

# Error on training set for 1 week ahead forecast
error_metrics(pred1, y_actual_train.iloc[-24*7:], 'SARIMAX(2,1,1)x(1,0,1,24) with Fourier terms 1 week', 
              test=False) 

############################################################################

# Prophet

X_trainP, X_testP, y_trainP, y_testP = train_test\
                           (lag_y[['y', 'population', 'temp', 'rain', 'hum', 'Kovo_match_on_match', 'week_weekday_1']], 
                           test_size=0.15, 
                           scale=False, #True
                           #cols_to_transform=cols_to_transform,
                           include_test_scale=False)

def data_prophet(X_train, X_test, y_train, y_test):
    data_train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    data_train = data_train.reset_index().rename(columns = {'Dates':'ds'})
    data_test = pd.merge(X_test, y_test, left_index=True, right_index=True)
    data_test  = data_test.reset_index().rename(columns = {'Dates':'ds'})
    return data_train, data_test

data_train, data_test = data_prophet(X_trainP, X_testP, y_trainP, y_testP)
data_train.tail(3)

# Importing Prophet
from prophet import Prophet

# Initiating fbprophet model; set the uncertainty interval to 95% (the Prophet default is 80%)
prop = Prophet(growth='linear', interval_width = 0.95, 
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto',
                seasonality_mode='additive',
                seasonality_prior_scale = 15
              )

prop.add_regressor('population', prior_scale=20, mode='additive', standardize=True)
prop.add_regressor('temp', prior_scale = 1, mode='additive', standardize=True)
prop.add_regressor('rain', prior_scale=10, mode='additive', standardize=True)
prop.add_regressor('hum', prior_scale=20, mode='additive', standardize=True)
prop.add_regressor('Kovo_match_on_match', prior_scale = 1, mode='additive', standardize='auto')
prop.add_regressor('week_weekday_1', prior_scale=10, mode='additive', standardize='auto')

prop.fit(data_train)

# Creating the dataframe with datetime values to predict on (making predictions on train as well as the test set)
future_dates = prop.make_future_dataframe(periods=len(data_test), freq='H', include_history=True)
# Aadding regressors 
future_dates = pd.merge(future_dates, (data_train.append(data_test)).drop('y', axis=1), on = 'ds')

forecast = prop.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#pd.plotting.register_matplotlib_converters()
_ = prop.plot(forecast, uncertainty = True, xlabel = 'Dates', ylabel = 'Water consumption MWH')
_ = prop.plot_components(forecast)
_ = plt.show()

plot_ts_pred(data_train.append(data_test).set_index('ds')['y'], forecast.set_index('ds')['yhat'], \
             model_name='FB Prophet with auto seasonality', og_ts_opacity = 0.5, pred_ts_opacity = 0.5)

# cross validation
from prophet.diagnostics import cross_validation
len(y_trainP)*2//3

df_cv = cross_validation(prop, initial='19855 hours', period='900 hours', horizon = '336 hours') # prophet cross validation 기간 설정
df_cv.head(3)

error_metrics(df_cv['yhat'], df_cv['y'], 'FB Prophet with auto seasonality 2 week ahead', test=False)
error_metrics(forecast.iloc[-len(data_test['y']):, ]['yhat'], data_test['y'], 
              'FB Prophet with auto seasonality', 
              test=True)

"""
Error metrics for model FB Prophet with auto seasonality 2 week ahead
RMSE or Root mean squared error: 128.99
Variance score: 0.42
Mean Absolute Error: 100.03
Mean Absolute Percentage Error: 86.33 %
>>> error_metrics(forecast.iloc[-len(data_test['y']):, ]['yhat'], data_test['y'],
...               'FB Prophet with auto seasonality', 
...               test=True)

Error metrics for model FB Prophet with auto seasonality
RMSE or Root mean squared error: 127.41
Variance score: 0.42
Mean Absolute Error: 96.41
Mean Absolute Percentage Error: 56.95 %
"""

############################################################################
## 7. Regression models using Fourier terms 
### Elastic Net

ycyc.head(2)

data = ycyc
cols_to_transform = ['population', 'temp', 'rain', 'hum', 'year']
l1_space = np.linspace(0, 1, 30)
alpha_space = np.logspace(-2, 0, 30)

X_trainF, X_testF, y_trainF, y_testF, elastic_net_opt_F = trend_model(data=data, cols_to_transform=cols_to_transform, 
                                                                l1_space=l1_space, alpha_space=alpha_space,
                                                                scale = True, test_size = 0.15, include_test_scale=False)


### Random Forest
random_grid['max_depth'] = [3,4,5,6,7,8]
random_grid

# First create the base model to tune
rf = RandomForestRegressor()
tscv = TimeSeriesSplit(n_splits=5)

# Random search of parameters
rfF = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               cv = tscv, verbose=2, random_state = 42, n_jobs = -1)

# Fit the random search model
rfF.fit(X_trainF, y_trainF)

rfF.best_params_
#rf.fit(X_train, y_train)

rfFourier = RandomForestRegressor(n_estimators= rfF.best_params_['n_estimators'], 
                                  max_features=rfF.best_params_['max_features'], 
                                  max_depth= rfF.best_params_['max_depth'], random_state = 42)
rfFourier.fit(X_trainF, y_trainF)

# comparison of scores using elastic net and RF

print("Training accuracy R2 using random forest is {}".format(rfF.score(X_trainF, y_trainF)))
print("Training accuracy R2 using Elastic net is {}".format(elastic_net_opt_F.score(X_trainF, y_trainF)))
print("\nTest accuracy R2 using random forest is {}".format(rfF.score(X_testF, y_testF)))
print("Test accuracy R2 using Elastic net is {}".format(elastic_net_opt_F.score(X_testF, y_testF)))

"""
>>> print("Training accuracy R2 using random forest is {}".format(rfF.score(X_trainF, y_trainF)))
Training accuracy R2 using random forest is 0.41999507736606545
>>> print("Training accuracy R2 using Elastic net is {}".format(elastic_net_opt_F.score(X_trainF, y_trainF)))
Training accuracy R2 using Elastic net is 0.37051716688992264
>>> print("\nTest accuracy R2 using random forest is {}".format(rfF.score(X_testF, y_testF)))
Test accuracy R2 using random forest is 0.4482461996843834
>>> print("Test accuracy R2 using Elastic net is {}".format(elastic_net_opt_F.score(X_testF, y_testF)))
Test accuracy R2 using Elastic net is 0.4354835380762875
"""

fig = go.Figure()

fig.add_trace(go.Scatter(x = X_testF.index, y = np.array(y_testF), name = "observed",
                         line_color = 'deepskyblue', opacity = 0.5))

fig.add_trace(go.Scatter(x = X_testF.index, y = rfF.predict(X_testF), name = "Random forest predictions",
                         line_color = 'lightslategrey', opacity = 0.5))

fig.add_trace(go.Scatter(x = X_testF.index, y = elastic_net_opt_F.predict(X_testF), name = "Elastic net predictions",
                         line_color = 'lightcoral', opacity = 0.5))

fig.update_layout(title_text = 'Observed test set vs Predicted energy using Random forest and elastic net reg on data with \
Fourier series',
                  xaxis_rangeslider_visible = True)
fig.show()

# error metrics
print('Random forest errors with fourier terms on training')
error_metrics(rfF.predict(X_trainF), y_trainF, model_name = 'Tuned Random forest with fourier terms', test=False)

"""
Error metrics for model Tuned Random forest with fourier terms
RMSE or Root mean squared error: 126.27
Variance score: 0.42
Mean Absolute Error: 96.59
Mean Absolute Percentage Error: 92.24 %
"""

print('\nRandom forest errors with fourier terms on test')
error_metrics(rfF.predict(X_testF), y_testF, model_name = 'Tuned Random forest with fourier terms', test=True)

"""
Error metrics for model Tuned Random forest with fourier terms
RMSE or Root mean squared error: 123.83
Variance score: 0.45
Mean Absolute Error: 94.94
Mean Absolute Percentage Error: 64.87 %
"""

print('\nElastic net errors with fourier terms on train')
error_metrics(elastic_net_opt_F.predict(X_trainF), y_trainF, model_name = 'Tuned Elastic net with fourier terms', test=False)

"""
Error metrics for model Tuned Elastic net with fourier terms
RMSE or Root mean squared error: 131.55
Variance score: 0.37
Mean Absolute Error: 99.03
Mean Absolute Percentage Error: 93.00 %
"""

print('\nElastic net errors with fourier terms on test')
error_metrics(elastic_net_opt_F.predict(X_testF), y_testF, model_name = 'Tuned Elastic net with fourier terms', test=True)

"""
Error metrics for model Tuned Elastic net with fourier terms
RMSE or Root mean squared error: 125.25
Variance score: 0.44
Mean Absolute Error: 98.84
Mean Absolute Percentage Error: 79.70 %
"""

## checking the feature importance for the random forest model
feat_imp = pd.DataFrame({'importance':rfFourier.feature_importances_})    
feat_imp['feature'] = X_trainF.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
#feat_imp = feat_imp.iloc[:top_n]
    
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
_ = feat_imp.plot.barh(title = 'Random Forest feature importance', figsize = (12,7))
_ = plt.show()
# 시간, 습도, 온도, 인구 정도가 좋은 feature

# # checking the feature importance for the Elastic net model

feat_imp = pd.DataFrame({'importance':np.abs(elastic_net_opt_F.coef_)})    
feat_imp['feature'] = X_trainF.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
#feat_imp = feat_imp.iloc[:top_n]
    
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
_ = feat_imp.plot.barh(title = 'Elastic net feature importance', figsize = (12,7))
_ = plt.show()

###########################################################################################
## XGBoost

import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 3, alpha = 5, n_estimators = 100, random_state=42)

xg_reg.fit(X_trainF, y_trainF)
preds_boost = xg_reg.predict(X_testF)
_ = error_metrics(preds_boost, y_testF, model_name='XGBoost with Fourier terms', test=True)
"""
Error metrics for model XGBoost with Fourier terms
RMSE or Root mean squared error: 117.86
Variance score: 0.50
Mean Absolute Error: 90.02
Mean Absolute Percentage Error: 56.05 %
"""

_ = error_metrics(xg_reg.predict(X_trainF), y_trainF, model_name='XGBoost with Fourier terms', test= False)
"""
Error metrics for model XGBoost with Fourier terms
RMSE or Root mean squared error: 121.27
Variance score: 0.46
Mean Absolute Error: 91.51
Mean Absolute Percentage Error: 73.97 %
"""

plot_ts_pred(y_trainF, xg_reg.predict(X_trainF), model_name='XGBoost with Fourier terms on Training set')

#pd.plotting.register_matplotlib_converters()
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [15, 7]
plt.show() # 인구, 습도, 온도, 시간, 주, 년도

## hyperparameter tuning
# Tuning the XGBoost model
xgbtuned = xgb.XGBRegressor()

param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'n_estimators': [10, 31, 52, 73, 94, 115, 136, 157, 178, 200]}

tscv = TimeSeriesSplit(n_splits=3)
xgbtunedreg = RandomizedSearchCV(xgbtuned, param_distributions=param_grid , 
                                   scoring='neg_mean_squared_error', n_iter=20, n_jobs=-1, 
                                   cv=tscv, verbose=2, random_state=42)

xgbtunedreg.fit(X_trainF, y_trainF)
best_score = xgbtunedreg.best_score_
best_params = xgbtunedreg.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))

preds_boost_tuned = xgbtunedreg.predict(X_testF)

"""
colsample_bylevel: 0.9
colsample_bytree: 0.9
gamma: 1.0
learning_rate: 0.2
max_depth: 4
min_child_weight: 3.0
n_estimators: 31
subsample: 0.9
"""

_ = error_metrics(preds_boost_tuned, y_testF, model_name='Tuned XGBoost with Fourier terms', test=True)
"""

Error metrics for model Tuned XGBoost with Fourier terms
RMSE or Root mean squared error: 117.38
Variance score: 0.50
Mean Absolute Error: 89.44
Mean Absolute Percentage Error: 53.14 %
"""

_ = error_metrics(xgbtunedreg.predict(X_trainF), y_trainF, model_name='Tuned XGBoost with Fourier terms', test=False)
"""
Error metrics for model Tuned XGBoost with Fourier terms
RMSE or Root mean squared error: 118.46
Variance score: 0.49
Mean Absolute Error: 89.77
Mean Absolute Percentage Error: 71.10 %
"""

plot_ts_pred(y_testF, preds_boost_tuned, model_name='Tuned XGBoost with Fourier terms on test')


###################################################################################################
## xgboost + prophet
#######################
fxdata = ycyc.copy()
fxdata.drop(['population', 'year'], axis = 1, inplace=True)

# Detrending the data
fxdata['y'] = fxdata['y'].to_numpy() - (forecast.trend + forecast.population).to_numpy()

fxdata['y'].isna().sum()

pd.plotting.register_matplotlib_converters()
fxdata['y'].plot()
plt.show()

data1 = fxdata
cols_to_transform = ['hum', 'rain', 'temp'] # technically we don't need to standardize data for tree based models

X_trainFP, X_testFP, y_trainFP, y_testFP = train_test(data=data1, cols_to_transform=cols_to_transform, 
                                                                scale = True, test_size = 0.15, include_test_scale=False)

# generating the model
xg_FP = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 5, n_estimators = 73, random_state=42)
# Fitting the model on the training set
xg_FP.fit(X_trainFP, y_trainFP)

# Predicting on test set
preds_boostFP = xg_FP.predict(X_testFP)
# adding back the trend
trend_and_PV = forecast.trend + forecast.population
boost_add_trend = preds_boostFP + trend_and_PV[-len(y_testF):]
plot_ts_pred(y_testF, boost_add_trend.to_numpy(), model_name='XGBoost with detrend, Fourier terms on test')

_ = error_metrics(boost_add_trend, y_testF, model_name='XGBoost with detrend Prophet, Fourier terms', test=True)
"""
Error metrics for model XGBoost with detrend Prophet, Fourier terms
RMSE or Root mean squared error: 125.65
Variance score: 0.43
Mean Absolute Error: 94.13
Mean Absolute Percentage Error: 51.30 %
"""

# on training set
preds_boostFP_train = xg_FP.predict(X_trainFP)
# adding back the trend
boost_add_trend_train = preds_boostFP_train + trend_and_PV[:(len(trend_and_PV) - len(y_testF))]
_ = error_metrics(boost_add_trend_train, y_trainF, model_name='XGBoost with detrend Prophet, Fourier terms', test=False)

"""
Error metrics for model XGBoost with detrend Prophet, Fourier terms
RMSE or Root mean squared error: 117.73
Variance score: 0.50
Mean Absolute Error: 89.57
Mean Absolute Percentage Error: 72.27 %
"""

trydf = pd.DataFrame.from_dict(dict_error)
#trydf.sort_values('MAPE', ascending=True).groupby(['model', 'train_test']).mean()
sorted_errors = trydf.pivot_table(index = 'model', columns = 'train_test', 
                                  aggfunc='min').sort_values(('MAPE', 'test'), ascending=True)
table = (sorted_errors.sort_index(axis=1, level=1, ascending=False).sort_index(axis=1, level=[0], sort_remaining=False)).\
round(3)
table.style.highlight_min(['MAPE', 'MAE', 'RMSE'], 
                                  axis=0).highlight_max(['R2'], axis=0).highlight_null(null_color='gray')

"""
                                                        MAE              MAPE              R2            RMSE
train_test                                            train     test    train     test  train   test    train     test
model
XGBoost with detrend Prophet, Fourier terms          89.574   94.127   72.267   51.303  0.496  0.432  117.727  125.647
Tuned XGBoost with Fourier terms                     89.770   89.437   71.100   53.136  0.490  0.504  118.459  117.382
Random forest with all lags                          91.632   89.997   75.882   55.703  0.490  0.498  118.389  118.165
XGBoost with Fourier terms                           91.508   90.016   73.966   56.054  0.465  0.500  121.275  117.862
FB Prophet with auto seasonality                        NaN   96.406      NaN   56.952    NaN  0.416      NaN  127.413
Elastic net with all lags                            92.185   88.826   79.783   62.794  0.475  0.532  120.178  114.095
Tuned Random forest with fourier terms               96.585   94.945   92.239   64.865  0.420  0.448  126.270  123.829
Simple linear regression with scaling                95.859   94.046   77.281   66.730  0.386  0.469  129.880  121.448
Ridge regression with scaling                        98.421   97.446   91.117   75.999  0.375  0.445  131.031  124.173
Tuned Elastic net with fourier terms                 99.028   98.836   93.003   79.696  0.371  0.435  131.546  125.253
Tuned Random forest with reduced hour space         109.805  106.240  120.935   84.054  0.273  0.343  141.288  135.129
Tuned elastic net regression with reduced hour ...  114.217  113.253  130.488  106.187  0.215  0.290  146.847  140.481
FB Prophet with auto seasonality 2 week ahead       100.035      NaN   86.328      NaN  0.423    NaN  128.985      NaN
"""

"""
Conclusion

- XGBoost 를 사용하는 것이 점수 향상에 가장 좋음 OR + Prophet을 이용한 detrend
"""