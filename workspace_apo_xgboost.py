####################
## import libraries
#####################
# importing all the required libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import copy
plt.style.use('bmh')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import random
import numpy as np
import random
import os

def my_seed_everywhere(seed: int = 42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os

my_seed = 42
my_seed_everywhere(my_seed)

import xgboost as xgb
import joblib
from sklearn.model_selection import RandomizedSearchCV

##################
## help function
###################
dict_error = dict()

def xgboost_cross_validation(train_df, test_df, params, n_iter=100) :

    xgbtuned = xgb.XGBRegressor()
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

# hour ??? night, morning, afternoon, evening?????? ??????
hour_dict = {'morning': list(np.arange(6,12)),'afternoon': list(np.arange(12,18)), 'evening': list(np.arange(18,24)),
            'night': [0, 1, 2, 3, 4, 5]}

time_dict = {'morning': list(np.arange(6,9)),'afternoon': list(np.arange(12,15)), 'evening': list(np.arange(18,22)),
            'night': [0, 1, 2, 3, 4, 5]}

# ??????
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

def add_feature(df) :

    df['rolling_6_mean'] = df['y'].shift(1).rolling(6).mean()
    df['rolling_12_mean'] = df['y'].shift(1).rolling(12).mean()
    df['rolling_24_mean'] = df['y'].shift(1).rolling(24).mean()

    df['rolling_6_std'] = df['y'].shift(1).rolling(6).std()
    df['rolling_12_std'] = df['y'].shift(1).rolling(12).std()
    df['rolling_24_std'] = df['y'].shift(1).rolling(24).std()

    return df

#####################
## data import
#####################
train_df = pd.read_csv('dataset/data_apo/task2_tr_st_1206.csv',index_col=0)
sample_df = pd.read_csv('dataset/data_apo/sample_apo.csv',index_col=0)
test_df_xgb = pd.read_csv('dataset/data_apo/task2_ts_st_1206.csv',index_col=0)

def season_calc(month):
    if month in [5,6,7,8]:
        return "summer"
    else:
        return "winter"

def is_season(ds) :
    date = pd.to_datetime(ds)
    return (date.month > 10 or date.month < 4)

def feature_preprocessing(train_df) :
    # 'ds' ???????????? ??????
    train_df.ds = pd.to_datetime(train_df['ds'])

    # ds ?????? ????????? feature ??????
    train_df['month'] = train_df.ds.dt.month
    train_df['year'] = train_df.ds.dt.year
    train_df['day'] = train_df.ds.dt.day
    train_df['hour'] = train_df.ds.dt.hour
    weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
    train_df['weekday'] = train_df.ds.dt.weekday.map(weekdays)

    train_df['dayofyear'] = train_df.ds.dt.dayofyear

    # ????????? ?????? feature ??????
    for iter in range(len(train_df)) :
        train_df.loc[iter,'season'] = season_calc(train_df.loc[iter, 'month'])

    # hour ??????
    train_df['time_of_day'] = train_df['hour'].apply(time_of_day)
    train_df['time'] = train_df['hour'].apply(time)

    # dtype ??????
    cat_cols = ['time_of_day','season','time','weekday','holiday','warning']  
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')

    # get_dummies
    train_df = pd.get_dummies(train_df, drop_first=True)

    return train_df

train_df = feature_preprocessing(train_df)
test_df_xgb = feature_preprocessing(test_df_xgb)

train_df.columns
test_df_xgb.columns

####################
## outlier + missing value
####################

# ????????? ?????? ??????
train_df.isna().sum() # 'y' : 8

# ??? ?????? ??????
train_df.describe() # ??????????????? ???????????? ??? ????????? ??????

# outlier??? missing value??? ??????
train_df.loc[(abs(train_df['y']) > 2000) | (train_df['y'] == 0), 'y'] = np.NaN # ???????????? missing value??? ?????? -> 153?????? missing value ??????

# index??? ds??? ??????
train_df.set_index('ds', inplace=True)
test_df_xgb.set_index('ds',inplace=True)

#train_df['y'].plot()
#plt.show()

# ffill()??? ????????? ?????????

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



################################################################################
train_df_final = train_df_Interpolation_time.copy()
test_df_final = test_df_xgb.copy()

param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        #'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'n_estimators': [10, 31, 52, 73, 94, 115, 136, 157, 178, 200]}

## lag variable ??????
for i in range(24):
    train_df_final['lag'+str(i+1)] = train_df_final['y'].shift(i+1)

for i in range(24):
    test_df_final['lag'+str(i+1)] = test_df_final['y'].shift(i+1)

train_df_lag = train_df_final.drop(['day','rain','time_evening','time_morning','time_night','dayofyear','weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
       'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday'], axis=1)
train_df_lag = train_df_lag.dropna()
test_df_lag = test_df_final.drop(['day','rain','time_evening','time_morning','time_night','dayofyear','weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
       'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday'], axis=1)

test_df_lag.index = pd.to_datetime(test_df_lag.index)
train_df_lag = add_fourier_terms(train_df_lag, year_k= 11, week_k=12, day_k=12)
test_df_lag = add_fourier_terms(test_df_lag, year_k= 11, week_k=12, day_k=12)

train_df_mul = add_feature(train_df_lag)
test_df_mul = add_feature(test_df_lag)
train_df_mul.dropna(inplace=True)

train_df_mul.columns
test_df_mul.columns

pred_y, X_test = xgboost_cross_validation(train_df_mul, test_df_mul, param_grid, n_iter=100)

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred_y[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

# result.to_csv('result/task2_now_best.csv')

########################################################

Best_params = {'subsample': 1.0, 'n_estimators': 73, 'min_child_weight': 0.5, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.25, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.8}

# {'subsample': 0.9, 'n_estimators': 157, 'min_child_weight': 0.5, 'max_depth': 9, 'learning_rate': 0.05, 'gamma': 1.0, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.7}
xgbtuned = xgb.XGBRegressor(**Best_params)

X_train = train_df_mul.drop('y', axis=1)
y_train = train_df_mul.y

X_test = test_df_mul.drop('y', axis=1)
y_test = test_df_mul.y

xgbtuned.fit(X_train, y_train)
pred = xgbtuned.predict(X_test)

mae = mean_absolute_error(y_test,pred)
print(mae)

# joblib.dump(xgbtuned, 'result/task2_best_now_2.pkl')

result = []
for iter in range(len(X_test)-336) :

    index = list(range(0+iter, 336+iter))
    mid = pred[index]
    result.append(mid)

result = np.array(result)
result = pd.DataFrame(result, index=X_test.index[:8425])
result.columns = sample_df.columns

result.to_csv('result/task2_best_2.csv')


