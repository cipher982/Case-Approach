# -*- coding: utf-8 -*-
"""
Created on Friday July 20 18:57 2018

@author: David Rose
"""

import time

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import lightgbm          as lgb

from sklearn                 import linear_model, model_selection, preprocessing
from sklearn.metrics         import mean_squared_error, r2_score, roc_curve, auc
from sklearn.model_selection import KFold

#pd.set_option('display.max_columns', 500)
#pd.set_option('display.max_rows', 500)

# Read in flat file
def read_data(file):
    df = pd.read_csv(file)
    return df

# Manually set data types to save time on larger files
def set_data_types(df):
    int_columns = ['Store_ID','Fiscal_Qtr','Fiscal_dayofWk','Hour']
    num_columns = ['AvgHourlyTemp','SalesRevenue']
    
    for col in int_columns:
        df[col] = df[col].astype(int)
    
    for col in num_columns:
        df[col] = df[col].astype(float)

    df['date'] = pd.to_datetime(df['DateStringYYYYMMDD'], format='%Y%m%d')
    df.drop('DateStringYYYYMMDD', axis=1, inplace=True)
        
    return df

# Some basic file preparation
def preprocess_data(df, trim_hours=False, clip_outliers=False):

	# Create a one-hot encoding to get rid of categorical features
    temp = pd.get_dummies(df, dummy_na=False)

    rows_before = temp.shape[0]

    # Filter out negative sales, assuming an error
    temp = temp[temp['SalesRevenue'] >= 0]
    
    # Filter out hours where business seems to be typically closed
    if trim_hours == True:
        temp = temp[temp['Hour'] >= 7]
        temp = temp[temp['Hour'] <= 20]
    
    # Trim down values too high/low
    if clip_outliers == True:
        temp = temp.apply(lambda col: col.clip(*col.quantile([0.05,0.95]).values))
        
    rows_after = temp.shape[0]
    print("Rows before:{0} after:{1}".format(rows_before, rows_after))
    return temp

# Main model function
def trainLGBM(df, targetFeature, store_ID, clip_outliers=False):

    # I like to create copys to be sure I neve alter the original file
    # This may need to be avoided with larger datasets 
    temp = df.copy()

    # Convert date to a numeric value
    temp['time'] = pd.to_timedelta(temp['date']).dt.total_seconds().astype(int)

    # Decided to remove some columns that just serve as proxies for others (daypart etc)
    keep_cols = ['Store_ID','Hour','date','time','HourlyWeather_clear-day','AvgHourlyTemp',
                 'HourlyWeather_clear-day', 'HourlyWeather_clear-night',
                 'HourlyWeather_cloudy', 'HourlyWeather_fog',
                 'HourlyWeather_partly-cloudy-day', 'HourlyWeather_partly-cloudy-night',
                 'HourlyWeather_rain', 'HourlyWeather_snow', 'HourlyWeather_wind','SalesRevenue']
    temp = temp[keep_cols]
    
    # Train a model specific to each store
    # Will work better with larger data
    temp = temp[temp['Store_ID'] == store_ID]
    #print("Shape for Store {0} = {1}".format(store_ID, size))

    # Trim store level outliers
    if clip_outliers == True:
    	temp = temp.apply(lambda col: col.clip(*col.quantile([0.05,0.95]).values))
    
    size = temp.shape[0]

    # Predict on last day of available data
    train_test_split_date = '2017-07-13'
    train = temp[temp['date'] <= train_test_split_date]
    test  = temp[temp['date'] >  train_test_split_date]
    
    #print("Train shape : {}".format(train.shape))
    #print("Test shape  : {}".format(test.shape))

    xDrop = ['SalesRevenue','date']
    
    X_train = train.drop(xDrop, axis=1, errors='ignore')
    y_train = train[targetFeature]
    
    X_test = test.drop(xDrop, axis=1, errors='ignore')
    y_test = test[targetFeature]
    
    #print("Training on {0}".format(targetFeature))
    
    #print("Initializing classifier. . .")
    gbm = lgb.LGBMRegressor(objective='regression',
                             metric='l2',
                             learning_rate=0.1,
                             #n_estimators=50,
                             #sub_feature=.514492,
                             #num_leaves=255,
                             #max_depth=7,
                             #min_data=16,
                             #verbosity=0,
                             #bagging_fraction=0.85,
                             #lambda_l1=.018953,
                             #lambda_l2=.05242,
                             #bagging_freq=5,
                             nthread=8,
                             silent=True)
    
    gbm.fit(X_train, y_train)
    #print("gbm.fit: X_train={0} y_Train={1}".format(np.shape(x_train), np.shape(y_train)))

    y_pred = gbm.predict(X_test)
    #print("r2_score: y_test={0} y_pred={1}".format(np.shape(y_test), np.shape(y_pred)))
    score = r2_score(y_test, y_pred)
    print("R2 Score is : {}".format(score))
    
    # Predict for specified criteria in PDF
    criteria = pd.DataFrame({'Store_ID' : 0, # Will be modified at prediction time
                             'Fiscal_Qtr' : 3,
                             'Fiscal_dayofWk' : 6,
                             'Hour' : 12, 
                             'AvgHourlyTemp' : 86,
                             'Daypart_Afternoon' : 0,
                             'Daypart_Breakfast' : 0,
                             'Daypart_Dinner' : 0,
                             'Daypart_Late Night' : 0, 
                             'Daypart_Lunch' : 1,
                             'HourlyWeather_clear-day' : 1,
                             'HourlyWeather_clear-night' : 0, 
                             'HourlyWeather_cloudy' : 0,
                             'HourlyWeather_fog' : 0, 
                             'HourlyWeather_partly-cloudy-day' : 0,
                             'HourlyWeather_partly-cloudy-night' : 0,
                             'HourlyWeather_rain' : 0,
                             'HourlyWeather_snow' : 0,
                             'HourlyWeather_wind' : 0, 
                             'time' : 1500076800, # Timedelta from specified criteria date
                             'date' : 0,
                             'SalesRevenue' : 0}, index=[0])
    criteria = criteria[keep_cols]
    criteria.drop(['date','SalesRevenue'], axis=1, errors='ignore', inplace=True)

    prediction = gbm.predict(criteria)

    return score, y_pred, gbm, prediction, size
    

def train_predict():
	df = set_data_types(read_data('data/SalesByHour.csv'))
	df = preprocess_data(df, trim_hours=True, clip_outliers=False)

	IDs         = []
	predictions = []
	sizes       = []
	r2s         = []
	storeIDs = np.unique(df['Store_ID'])
	# Loop through each store_ID and fit/predict
	for ID in storeIDs:
	    IDs.append(ID)
	    score, y_pred, gbm, prediction, size = trainLGBM(df, 'SalesRevenue', ID, clip_outliers=False)
	    sizes.append(size)
	    predictions.append(prediction[0])
	    r2s.append(score)
	
	# Create a nice and clean metrics dataset    
	predictions_df = pd.DataFrame({'Store_ID'           : IDs,
	                               'Size'               : sizes,
	                               'R2_Score'           : r2s,
	                               'Revenue_Prediction' : predictions}).set_index('Store_ID')

	return predictions_df

# Code starts running from here	    
if __name__ == '__main__':

	csv_path = "LGBM_Store_Predictions_{0}.csv".format(int(time.time()))
	store_predictions = train_predict()
	store_predictions.to_csv(csv_path) # Save results to working directory
	print(store_predictions)
	print("Median R2 Score:{0}".format(np.median(store_predictions['R2_Score'])))
	print("Predictions output to {0}".format(csv_path))
