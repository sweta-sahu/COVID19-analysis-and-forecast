#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd


# # Import data

# In[2]:


cases_per_state = pd.read_csv('india_cases_per_country.csv')


# In[3]:


global_confirmed_cases = pd.read_csv('time_series_covid19_confirmed_global.csv')


# In[4]:


global_death_cases = pd.read_csv('time_series_covid19_deaths_global.csv')


# In[5]:


global_recovered_cases = pd.read_csv('time_series_covid19_recovered_global.csv')


# In[6]:


global_confirmed_cases.head()


# # Removing uneccesary data

# In[7]:


global_confirmed_cases.drop(['Lat', 'Long','Province/State'],axis=1, inplace=True)


# In[8]:


global_confirmed_cases.head()


# In[9]:


global_death_cases.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)


# In[10]:


global_recovered_cases.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)


# # Analysing state wise data in India

# In[11]:


cases_per_state.drop('S. No.', axis=1, inplace=True)


# In[12]:


cases_per_state.drop(36, axis=0, inplace=True)


# In[13]:


cases_per_state.sort_values('Total Confirmed cases', ascending=False, inplace=True)


# In[14]:


cases_per_state.style.background_gradient(cmap='Reds')


# In[15]:


sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(12,8))
data = cases_per_state[['Name of State / UT', 'Cured/Discharged/Migrated', 'Deaths', 'Total Confirmed cases']]
sns.set_color_codes('pastel')
sns.barplot(x='Total Confirmed cases', y='Name of State / UT', data=data, label='Total', color='r')
sns.set_color_codes('muted')
sns.barplot(x='Cured/Discharged/Migrated', y='Name of State / UT', data=data, label='Cured', color='r')

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 175000), ylabel = "", xlabel = 'Cases')
sns.despine(left=True, bottom=True)


# # Global data Analysis

# In[16]:


latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/06-30-2020.csv')


# In[17]:


latest_data.head()


# In[18]:


latest_data.drop(['Admin2', 'Province_State', 'Last_Update','Lat', 'Long_', 'Combined_Key', 'FIPS'], axis=1, inplace=True)


# In[19]:


latest_data = latest_data.groupby('Country_Region').sum()


# In[20]:


latest_data.sort_values('Confirmed', ascending=False, inplace=True)
latest_data


# ## Top 10 countries ranked according to confirmed cases

# In[21]:


latest_data.head(10).style.background_gradient(cmap='Blues')


# ## Comparing India with US, Italy and China

# In[22]:


cols = global_confirmed_cases.keys()
confirmed = global_confirmed_cases.loc[:, cols[1]:cols[-1]]
deaths = global_death_cases.loc[:, cols[1]:cols[-1]]
recoveries = global_recovered_cases.loc[:, cols[1]:cols[-1]]
confirmed


# In[23]:


dates = confirmed.keys()

world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 

china_cases = [] 
italy_cases = []
us_cases = [] 
india_cases = []

china_deaths = [] 
italy_deaths = []
us_deaths = [] 
india_deaths = []


china_recoveries = [] 
italy_recoveries = []
us_recoveries = [] 
india_recoveries = []


# In[24]:


for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    
    
    china_cases.append(global_confirmed_cases[global_confirmed_cases['Country/Region']=='China'][i].sum())
    italy_cases.append(global_confirmed_cases[global_confirmed_cases['Country/Region']=='Italy'][i].sum())
    us_cases.append(global_confirmed_cases[global_confirmed_cases['Country/Region']=='US'][i].sum())
    india_cases.append(global_confirmed_cases[global_confirmed_cases['Country/Region']=='India'][i].sum())
    
    
    china_deaths.append(global_death_cases[global_death_cases['Country/Region']=='China'][i].sum())
    italy_deaths.append(global_death_cases[global_death_cases['Country/Region']=='Italy'][i].sum())
    us_deaths.append(global_death_cases[global_death_cases['Country/Region']=='US'][i].sum())
    india_deaths.append(global_death_cases[global_death_cases['Country/Region']=='India'][i].sum())
    
    
    china_recoveries.append(global_recovered_cases[global_recovered_cases['Country/Region']=='China'][i].sum())
    italy_recoveries.append(global_recovered_cases[global_recovered_cases['Country/Region']=='Italy'][i].sum())
    us_recoveries.append(global_recovered_cases[global_recovered_cases['Country/Region']=='US'][i].sum())
    india_recoveries.append(global_recovered_cases[global_recovered_cases['Country/Region']=='India'][i].sum())


# In[25]:


world_cases


# In[26]:


india_cases


# In[27]:


f, ax = plt.subplots(figsize=(12,8))
data_india = pd.DataFrame({'dates': dates, 'cases': india_cases})
sns.barplot(x='dates', y='cases', data=data_india)
sns.set(style="whitegrid")
ax.set(xlabel = "Dates from 1/22/20 to 6/30/20", ylabel = 'Cases', title="Trend of confirmed cases in India")


# In[28]:


f, ax = plt.subplots(figsize=(12,8))
data_china = pd.DataFrame({'dates': dates, 'cases': china_cases})
sns.barplot(x='dates', y='cases', data=data_china)
sns.set(style="whitegrid")
ax.set(xlabel = "Dates from 1/22/20 to 6/30/20", ylabel = 'Cases', title="Trend of confirmed cases in China")


# In[29]:


f, ax = plt.subplots(figsize=(12,8))
data_us = pd.DataFrame({'dates': dates, 'cases': us_cases})
sns.barplot(x='dates', y='cases', data=data_us)
sns.set(style="whitegrid")
ax.set(xlabel = "Dates from 1/22/20 to 6/30/20", ylabel = 'Cases', title="Trend of confirmed cases in US")


# In[30]:


f, ax = plt.subplots(figsize=(12,8))
data_italy = pd.DataFrame({'dates': dates, 'cases': italy_cases})
sns.barplot(x='dates', y='cases', data=data_italy)
sns.set(style="whitegrid")
ax.set(xlabel = "Dates from 1/22/20 to 6/30/20", ylabel = 'Cases', title="Trend of confirmed cases in Italy")


# In[31]:


confirmed_cases = global_confirmed_cases.groupby('Country/Region').sum()
death_cases = global_death_cases.groupby('Country/Region').sum()
recovered_cases = global_recovered_cases.groupby('Country/Region').sum()


# In[32]:


fig = plt.figure(figsize=(12, 8))
confirmed_cases.loc['India'].plot()
confirmed_cases.loc['China'].plot()
confirmed_cases.loc['US'].plot()
confirmed_cases.loc['Italy'].plot()
plt.legend()
plt.title('India vs US vs China vs Italy - Confirmed Cases')


# In[33]:


fig = plt.figure(figsize=(12, 8))
death_cases.loc['India'].plot()
death_cases.loc['China'].plot()
death_cases.loc['US'].plot()
death_cases.loc['Italy'].plot()
plt.legend()
plt.title('India vs US vs China vs Italy - Deaths Reported')


# In[34]:


fig = plt.figure(figsize=(12, 8))
recovered_cases.loc['India'].plot()
recovered_cases.loc['China'].plot()
recovered_cases.loc['US'].plot()
recovered_cases.loc['Italy'].plot()
plt.legend()
plt.title('India vs US vs China vs Italy - Recovered Cases')


# # Predicting future values

# In[35]:


import numpy as np 
import random
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 


# In[36]:


def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 


# In[37]:


# confirmed cases
world_daily_increase = daily_increase(world_cases)
china_daily_increase = daily_increase(china_cases)
italy_daily_increase = daily_increase(italy_cases)
us_daily_increase = daily_increase(us_cases)
india_daily_increase = daily_increase(india_cases)


# In[38]:


world_daily_death = daily_increase(total_deaths)
china_daily_death = daily_increase(china_deaths)
italy_daily_death = daily_increase(italy_deaths)
us_daily_death = daily_increase(us_deaths)
india_daily_death = daily_increase(india_deaths)


# In[39]:


# recoveries
world_daily_recovery = daily_increase(total_recovered)
china_daily_recovery = daily_increase(china_recoveries)
italy_daily_recovery = daily_increase(italy_recoveries)
us_daily_recovery = daily_increase(us_recoveries)
india_daily_recovery = daily_increase(india_recoveries)


# In[40]:


days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
india_cases = np.array(india_cases).reshape(-1, 1)
india_deaths = np.array(india_deaths).reshape(-1, 1)
india_recovered = np.array(india_recoveries).reshape(-1, 1)


# In[41]:


days_in_future = 30
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forecast[:-30]


# In[42]:


future_forecast


# In[43]:


start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# In[44]:


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, india_cases, test_size=0.05, shuffle=False) 


# In[45]:


# transform our data for polynomial regression
poly = PolynomialFeatures(degree=4)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forecast = poly.fit_transform(future_forecast)


# In[46]:


# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forecast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))


# In[47]:


linear_pred = linear_pred.reshape(1,-1)[0]
poly_df = pd.DataFrame({'Date': future_forecast_dates[-30:], 'Predicted number of Confirmed Cases in India': np.round(linear_pred[-30:])})
poly_df


# In[48]:


from fbprophet import Prophet


# In[49]:


cases = []
for i in range(0,161,1):
    cases.append(india_cases[i][0])
india = pd.DataFrame({'ds': dates, 'y': cases})
india.tail()


# In[50]:


m = Prophet(interval_width=0.95)
m.fit(india)
future = m.make_future_dataframe(periods=7)
future.tail(7)


# In[51]:


#predicting the future with date, and upper and lower limit of y value
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)


# In[52]:


confirmed_forecast_plot = m.plot(forecast)


# In[53]:


confirmed_forecast_plot =m.plot_components(forecast)


# In[ ]:




