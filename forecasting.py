#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 07:45:08 2020

@author: rajkumar
"""

#importing required libraries 
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
from warnings import filterwarnings
filterwarnings('ignore')
#%matplotlib inline
#importing Dataset
df = pd.read_excel("D:\\My Data\\Downloads\\project_forecating_final_code.xlsx")

#############################  APPLE      ###############################

Apple = df.loc[df['Commodity'] == 'Apple']

#checking max date and min date from Dates column(Price Date)

Apple['Price Date'].min(), Apple['Price Date'].max()
Apple.columns

#dropping unwanted columns from apple dataset

cols = ['Sl no.', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']
Apple.drop(cols, axis=1, inplace=True)

#renaming Apple columns

Apple.rename(columns = {'Modal Price (Rs./Quintal)':'ModalPrice'}, inplace = True)
Apple.rename(columns = {'Price Date':'PriceDate'}, inplace = True)

#sorting the date column 
Apple = Apple.sort_values('PriceDate')
#checking null values
Apple.isnull().sum() #no null values found

#grouping columns with index
Apple = Apple.groupby('PriceDate')['ModalPrice'].sum().reset_index()

#indexing with timeseries data
Apple = Apple.set_index('PriceDate')
Apple.index

#Our current datetime data can be tricky to work ,
#we will use the averages daily Prices value for that month instead, 
#and we are using the start of each month as the timestamp.
y = Apple['ModalPrice'].resample('MS').mean()
y['2020':]

#Visualizing Average(Modal) Prices Time Series Data
y.plot(figsize=(15, 6))
plt.show()

#decomposition seasonality
from pylab import rcParams
rcParams['figure.figsize'] = 18, 6
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#performing ARIMA Model
#for examples of parameter combinations for Seasonal ARIMA  
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


#for parameters Selection
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:    
            Continue

#we see that lowest AIC value at ARIMA(1,1,1)*(1,1,0,12)      
#fitting ARIMA model
mod = sm.tsa.statespace.SARIMAX(y,order=(1, 1, 1),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False,enforce_invertibility=False)
                                
results = mod.fit()
print(results.summary().tables[1])            

#Checking Unusual Behaviour by running model Diagnostics              
results.plot_diagnostics(figsize=(16, 6))
plt.show()
#our model diagnostics suggests that the model residuals are near normally distributed.

#we compare predicted prices to real prices of the time series, 
#and we set forecasts to start at 2020–01–01 to the end of the data.

#validating forecasting
pred = results.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Apple Prices')
plt.legend()
plt.show()
                                            
y_forecasted = pred.predicted_mean
y_truth = y['2020-01-01':]

#MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

Apple_Mape=MAPE(y_forecasted,y_truth)


mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#The Mean Squared Error of our forecasts is 210805406.76

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
#The Root Mean Squared Error of our forecasts is 14519.14

#Producing and visualizing forecasts

pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Apple Prices')
plt.legend()
plt.show()

#our model clearly captures apples prices Seasonality

########################    BANANA   ###########################################

#taking banana commodity from fruits(df) dataset
Banana= df.loc[df['Commodity'] == 'Banana']

##checking max date and min date from Dates column(Price Date)
Banana['Price Date'].min(), Banana['Price Date'].max()
#checking Columns
Banana.columns

##dropping unwanted columns from banana dataset
cols = ['Sl no.', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']
Banana.drop(cols, axis=1, inplace=True)
#renaming columns
Banana.rename(columns = {'Modal Price (Rs./Quintal)':'ModalPrice'}, inplace = True)
Banana.rename(columns = {'Price Date':'PriceDate'}, inplace = True)
#sorting PriceDate
Banana = Banana.sort_values('PriceDate')
#removing null values
Banana.isnull().sum()

Banana = Banana.groupby('PriceDate')['ModalPrice'].sum().reset_index()

#indexing with timeseries data
Banana= Banana.set_index('PriceDate')
Banana.index

#Our current datetime data can be tricky to work ,
#we will use the averages daily Prices value for that month instead, 
#and we are using the start of each month as the timestamp.
y = Banana['ModalPrice'].resample('MS').mean()
y['2020':]

#Visualizing Average(Modal) Prices Time Series Data
y.plot(figsize=(15, 6))
plt.show()

#decomposition seasonality
from pylab import rcParams
rcParams['figure.figsize'] = 18, 6
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#performing ARIMA Model
#for examples of parameter combinations for Seasonal ARIMA  
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#for parameters Selection
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:    
            Continue

#we see that lowest AIC value at ARIMA(1,1,1)*(1,1,0,12) but it is not making p value lesser
#so we taking ARIMA(1, 1, 0)x(1, 1, 0, 12)12 - AIC:655.9322908317694 instead          
#fitting ARIMA model
mod = sm.tsa.statespace.SARIMAX(y,order=(0, 1, 0),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False,enforce_invertibility=False)
                                
results = mod.fit()
print(results.summary().tables[1])            

#Checking Unusual Behaviour by running model Diagnostics              
results.plot_diagnostics(figsize=(16, 6))
plt.show()
#our model diagnostics suggests that the model residuals are near normally distributed.

#we compare predicted prices to real prices of the time series, 
#and we set forecasts to start at 2020–01–01 to the end of the data.

#validating forecasting
pred = results.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Banana Prices')
plt.legend()
plt.show()
                                            
y_forecasted = pred.predicted_mean
y_truth = y['2020-01-01':]

#MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

Banana_Mape=MAPE(y_forecasted,y_truth)


mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#The Mean Squared Error of our forecasts is 2282376.82

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
#The Root Mean Squared Error of our forecasts is 1510.75

#Producing and visualizing forecasts

pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Banana Prices')
plt.legend()
plt.show()

#our model clearly captures Banana prices Seasonality

#################### LEMONS #################


Lemon= df.loc[df['Commodity'] == 'Lemon']

Lemon['Price Date'].min(), Lemon['Price Date'].max()
Lemon.columns
cols = ['Sl no.', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']
Lemon.drop(cols, axis=1, inplace=True)
Lemon.rename(columns = {'Modal Price (Rs./Quintal)':'ModalPrice'}, inplace = True)
Lemon.rename(columns = {'Price Date':'PriceDate'}, inplace = True)

Lemon = Lemon.sort_values('PriceDate')
Lemon.isnull().sum()

Lemon = Lemon.groupby('PriceDate')['ModalPrice'].sum().reset_index()

#indexing with timeseries data
Lemon= Lemon.set_index('PriceDate')
Lemon.index

#Our current datetime data can be tricky to work ,
#we will use the averages daily Prices value for that month instead, 
#and we are using the start of each month as the timestamp.
y = Lemon['ModalPrice'].resample('MS').mean()
y['2020':]

#Visualizing Average(Modal) Prices Time Series Data
y.plot(figsize=(15, 6))
plt.show()

#decomposition seasonality
from pylab import rcParams
rcParams['figure.figsize'] = 18, 6
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#performing ARIMA Model
#for examples of parameter combinations for Seasonal ARIMA  
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#for parameters Selection
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:    
            Continue

#we see that lowest AIC value at ARIMA(1,1,1)*(1,1,1,12) but it is not generating good p value
#so we decided to choose ARIMA(1, 0, 0)x(0, 1, 1, 12)12 - AIC:247.33229760253002           
#fitting ARIMA model
mod = sm.tsa.statespace.SARIMAX(y,order=(1, 0, 0),seasonal_order=(0, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)
                                
results = mod.fit()
print(results.summary().tables[1])            

#Checking Unusual Behaviour by running model Diagnostics              
results.plot_diagnostics(figsize=(16, 6))
plt.show()
#our model diagnostics suggests that the model residuals are near normally distributed.

#we compare predicted prices to real prices of the time series, 
#and we set forecasts to start at 2020–01–01 to the end of the data.

#validating forecasting
pred = results.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Lemon Prices')
plt.legend()
plt.show()
                                            
y_forecasted = pred.predicted_mean
y_truth = y['2020-01-01':]

#MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

Lemon_Mape=MAPE(y_forecasted,y_truth)


mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#The Mean Squared Error of our forecasts is 19860993.26

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
#The Root Mean Squared Error of our forecasts is 4456.57

#Producing and visualizing forecasts

pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Lemon Prices')
plt.legend()
plt.show()

#our model clearly captures apples prices Seasonality


###############################Vegetables#######################################

#############################CARROT#####################################

Carrot = df.loc[df['Commodity'] == 'Carrot']

Carrot['Price Date'].min(), Carrot['Price Date'].max()
Carrot.columns
cols = ['Sl no.', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']
Carrot.drop(cols, axis=1, inplace=True)
Carrot.rename(columns = {'Modal Price (Rs./Quintal)':'ModalPrice'}, inplace = True)
Carrot.rename(columns = {'Price Date':'PriceDate'}, inplace = True)

Carrot = Carrot.sort_values('PriceDate')
Carrot.isnull().sum()

Carrot = Carrot.groupby('PriceDate')['ModalPrice'].sum().reset_index()

#indexing with timeseries data
Carrot = Carrot.set_index('PriceDate')
Carrot.index

#Our current datetime data can be tricky to work 
y = Carrot['ModalPrice'].resample('MS').mean()
y['2020':]

y.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 6
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#seasonality
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:    
            Continue
 #we take the lower AIC value 
 #ARIMA(0, 0, 1)x(1, 1, 1, 12)12 - AIC:202.007302825275         
mod = sm.tsa.statespace.SARIMAX(y,order=(0, 0, 1),seasonal_order=(1, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)
                                
results = mod.fit()
print(results.summary().tables[1])            
                 
results.plot_diagnostics(figsize=(16, 6))
plt.show()

#validating forecasting
pred = results.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Carrot Prices')
plt.legend()
plt.show()

#predicting mean                                         
y_forecasted = pred.predicted_mean
y_truth = y['2020-01-01':]

#MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

Carrot_Mape=MAPE(y_forecasted,y_truth)



#MSE
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#RMSE
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

#producing and visualizing forecasting
pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Carrot Prices')
plt.legend()
plt.show()

######################## BEETROOT ##########################################
Beetroot = df.loc[df['Commodity'] == 'Beetroot']

Beetroot['Price Date'].min(), Beetroot['Price Date'].max()
Beetroot.columns
cols = ['Sl no.', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']
Beetroot.drop(cols, axis=1, inplace=True)
Beetroot.rename(columns = {'Modal Price (Rs./Quintal)':'ModalPrice'}, inplace = True)
Beetroot.rename(columns = {'Price Date':'PriceDate'}, inplace = True)

Beetroot = Beetroot.sort_values('PriceDate')
Beetroot.isnull().sum()

Beetroot = Beetroot.groupby('PriceDate')['ModalPrice'].sum().reset_index()

#indexing with timeseries data
Beetroot = Beetroot.set_index('PriceDate')
Beetroot.index

#Our current datetime data can be tricky to work 
y = Beetroot['ModalPrice'].resample('MS').mean()
y['2020':]

y.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18,6
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#seasonality
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:    
            Continue
 #we take the lower AIC value 
 #ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:610.7764323607779     
mod = sm.tsa.statespace.SARIMAX(y,order=(1, 1, 1),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False,enforce_invertibility=False)
                                
results = mod.fit()
print(results.summary().tables[1])            
                 
results.plot_diagnostics(figsize=(16, 6))
plt.show()

#validating forecasting
pred = results.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Beetroot Prices')
plt.legend()
plt.show()

#predicting mean                                         
y_forecasted = pred.predicted_mean
y_truth = y['2020-01-01':]

#MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

BeetRoot_Mape=MAPE(y_forecasted,y_truth)


#MSE
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#The Mean Squared Error of our forecasts is 2889276.44

#RMSE
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
#The Root Mean Squared Error of our forecasts is 1699.79

#producing and visualizing forecasting
pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Beetroot Prices')
plt.legend()
plt.show()

#########################CABBAGE#############################
Cabbage = df.loc[df['Commodity'] == 'Cabbage']

Cabbage['Price Date'].min(), Cabbage['Price Date'].max()
Cabbage.columns
cols = ['Sl no.', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']
Cabbage.drop(cols, axis=1, inplace=True)
Cabbage.rename(columns = {'Modal Price (Rs./Quintal)':'ModalPrice'}, inplace = True)
Cabbage.rename(columns = {'Price Date':'PriceDate'}, inplace = True)

Cabbage = Cabbage.sort_values('PriceDate')
Cabbage.isnull().sum()

Cabbage = Cabbage.groupby('PriceDate')['ModalPrice'].sum().reset_index()

#indexing with timeseries data
Cabbage = Cabbage.set_index('PriceDate')
Cabbage.index

#Our current datetime data can be tricky to work 
y = Cabbage['ModalPrice'].resample('MS').mean()
y['2020':]

y.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 6
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#seasonality
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:    
            Continue
 #we take the lower AIC value  
##ARIMA(1, 1, 0)x(1, 1, 0, 12)12 - AIC:196.24842814918154
  
mod = sm.tsa.statespace.SARIMAX(y,order=(1, 1, 0),seasonal_order=(1, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)
                                
results = mod.fit()
print(results.summary().tables[1])            
                 
results.plot_diagnostics(figsize=(16, 6))
plt.show()

#validating forecasting
pred = results.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Cabbage Prices')
plt.legend()
plt.show()

#predicting mean                                         
y_forecasted = pred.predicted_mean
y_truth = y['2020-01-01':]

#MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

Cabbage_Mape=MAPE(y_forecasted,y_truth)



#MSE
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#The Mean Squared Error of our forecasts is 1233590.41

#RMSE
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
#The Root Mean Squared Error of our forecasts is  1110.67

#producing and visualizing forecasting
pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Cabbage Prices')
plt.legend()
plt.show()

