import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Read
df = pd.read_csv('dataset/Pedestrian_Counting_System_-_Monthly__counts_per_hour_.csv', usecols=['Date_Time', 'Sensor_Name', 'Hourly_Counts'])

df.head()

# Convert date to datetime
df['Date_Time'] = pd.to_datetime(df['Date_Time'])

# Group all sensors
df_grouped = df.groupby(['Date_Time']).agg({'Hourly_Counts': 'sum'}).rename(columns={'Hourly_Counts': 'Count_Pedestrians'})

# Aggregate weekly
df_weekly = df_grouped.resample('W').sum()

# Filter from the start of 2010 to end of 2019
df_weekly = df_weekly['2010-01-01': '2019-12-31']

df_weekly.index

df_weekly

y = df_weekly.squeeze() # prepare the data as a pandas Series

from sktime.forecasting.model_selection import temporal_train_test_split

y_train, y_test = temporal_train_test_split(y, test_size=26) # Predict from 1st July 2019

from sktime.forecasting.base import ForecastingHorizon

fh = ForecastingHorizon(y_test.index, is_relative=False)

from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction

regressor = LinearRegression()
forecaster = make_reduction(regressor, window_length=52, strategy="recursive")

forecaster.fit(y_train)

y_pred = forecaster.predict(fh)

from xgboost import XGBRegressor

regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
forecaster = make_reduction(regressor, window_length=52, strategy="recursive")

# Create an exogenous dataframe indicating the month
X = pd.DataFrame({'month': y.index.month}, index=y.index)
X = pd.get_dummies(X.astype(str), drop_first=True)
X_train, X_test = temporal_train_test_split(X, test_size=26) # Predict from 1st July 2019

# Fit
forecaster.fit(y=y_train, X=X_train)

y_train
X_train

# Predict
y_pred = forecaster.predict(fh=fh, X=X_test)

from sktime.forecasting.model_selection import SlidingWindowSplitter

validation_size = 26
# cv = SlidingWindowSplitter(window_length=400, fh=validation_size)

fh = list(range(1,27))


from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.model_selection import ForecastingGridSearchCV

param_grid = {
    'estimator__max_depth': [3, 5, 6, 10, 15, 20],
    'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'estimator__subsample': np.arange(0.5, 1.0, 0.1),
    'estimator__colsample_bytree': np.arange(0.4, 1.0, 0.1),
    'estimator__colsample_bylevel': np.arange(0.4, 1.0, 0.1),
    'estimator__n_estimators': [100, 500, 1000]
}

regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
forecaster = make_reduction(regressor, window_length=52, strategy="recursive")

gscv = ForecastingRandomizedSearchCV(forecaster, cv=SlidingWindowSplitter(window_length= 450, fh=fh), param_distributions=param_grid, n_iter=100, random_state=42)

gscv.fit(y=y_train, X=X_train) 
gscv.cv_results_