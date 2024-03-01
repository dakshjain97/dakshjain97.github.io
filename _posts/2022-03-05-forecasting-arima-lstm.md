---
layout: post
title: Forecasting restaurant visitors using ARIMA, SARIMA & LSTM
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [Forecasting, ARIMA, SARIMA, LSTM]
author: Daksh Jain
---
In this notebooks objective is to predict number no of visitors across ~800 restaurants in JAPAN, first data exploration & data cleaning is done to identify certain patterns then for POC purpose forecasting is done only for 1 restaurant (but same can be replicated across all restaurants)

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/recruit-restaurant-visitor-forecasting/air_reserve.csv.zip
    /kaggle/input/recruit-restaurant-visitor-forecasting/hpg_store_info.csv.zip
    /kaggle/input/recruit-restaurant-visitor-forecasting/hpg_reserve.csv.zip
    /kaggle/input/recruit-restaurant-visitor-forecasting/sample_submission.csv.zip
    /kaggle/input/recruit-restaurant-visitor-forecasting/air_visit_data.csv.zip
    /kaggle/input/recruit-restaurant-visitor-forecasting/air_store_info.csv.zip
    /kaggle/input/recruit-restaurant-visitor-forecasting/date_info.csv.zip
    /kaggle/input/recruit-restaurant-visitor-forecasting/store_id_relation.csv.zip



```python
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
```


```python
#Loading datasets
df_air_visit= pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/air_visit_data.csv.zip')
df_air_store_info= pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/air_store_info.csv.zip')
df_date_info= pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/date_info.csv.zip')
```


```python
df_date_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>calendar_date</th>
      <th>day_of_week</th>
      <th>holiday_flg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>Friday</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-02</td>
      <td>Saturday</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-03</td>
      <td>Sunday</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-04</td>
      <td>Monday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-05</td>
      <td>Tuesday</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_air_store_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>air_store_id</th>
      <th>air_genre_name</th>
      <th>air_area_name</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>air_0f0cdeee6c9bf3d7</td>
      <td>Italian/French</td>
      <td>Hyōgo-ken Kōbe-shi Kumoidōri</td>
      <td>34.695124</td>
      <td>135.197853</td>
    </tr>
    <tr>
      <th>1</th>
      <td>air_7cc17a324ae5c7dc</td>
      <td>Italian/French</td>
      <td>Hyōgo-ken Kōbe-shi Kumoidōri</td>
      <td>34.695124</td>
      <td>135.197853</td>
    </tr>
    <tr>
      <th>2</th>
      <td>air_fee8dcf4d619598e</td>
      <td>Italian/French</td>
      <td>Hyōgo-ken Kōbe-shi Kumoidōri</td>
      <td>34.695124</td>
      <td>135.197853</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air_a17f0778617c76e2</td>
      <td>Italian/French</td>
      <td>Hyōgo-ken Kōbe-shi Kumoidōri</td>
      <td>34.695124</td>
      <td>135.197853</td>
    </tr>
    <tr>
      <th>4</th>
      <td>air_83db5aff8f50478e</td>
      <td>Italian/French</td>
      <td>Tōkyō-to Minato-ku Shibakōen</td>
      <td>35.658068</td>
      <td>139.751599</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_air_visit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>air_store_id</th>
      <th>visit_date</th>
      <th>visitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-13</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-14</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-15</td>
      <td>29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-16</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-18</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



# Data Exploration & Cleaning


```python
df_air_visit[df_air_visit.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>air_store_id</th>
      <th>visit_date</th>
      <th>visitors</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df_air_visit.shape
```




    (252108, 3)




```python
df_air_visit.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 252108 entries, 0 to 252107
    Data columns (total 3 columns):
     #   Column        Non-Null Count   Dtype 
    ---  ------        --------------   ----- 
     0   air_store_id  252108 non-null  object
     1   visit_date    252108 non-null  object
     2   visitors      252108 non-null  int64 
    dtypes: int64(1), object(2)
    memory usage: 5.8+ MB



```python
#Total 829 stores
stores = list(set(df_air_visit['air_store_id']))
```


```python
len(stores)
```




    829




```python
#Plot for a single store id
df_air_visit[(df_air_visit['air_store_id']=='air_ba937bf13d40fb24')].plot('visit_date','visitors')
```




    <Axes: xlabel='visit_date'>




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_12_1.png)
    



```python
#Merging all data
df_all = pd.merge(pd.merge(df_air_visit,df_date_info,left_on = ['visit_date'],right_on = ['calendar_date'],how = 'inner'),
         df_air_store_info,on = ['air_store_id'])
```


```python
df_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>air_store_id</th>
      <th>visit_date</th>
      <th>visitors</th>
      <th>calendar_date</th>
      <th>day_of_week</th>
      <th>holiday_flg</th>
      <th>air_genre_name</th>
      <th>air_area_name</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-13</td>
      <td>25</td>
      <td>2016-01-13</td>
      <td>Wednesday</td>
      <td>0</td>
      <td>Dining bar</td>
      <td>Tōkyō-to Minato-ku Shibakōen</td>
      <td>35.658068</td>
      <td>139.751599</td>
    </tr>
    <tr>
      <th>1</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-14</td>
      <td>32</td>
      <td>2016-01-14</td>
      <td>Thursday</td>
      <td>0</td>
      <td>Dining bar</td>
      <td>Tōkyō-to Minato-ku Shibakōen</td>
      <td>35.658068</td>
      <td>139.751599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-15</td>
      <td>29</td>
      <td>2016-01-15</td>
      <td>Friday</td>
      <td>0</td>
      <td>Dining bar</td>
      <td>Tōkyō-to Minato-ku Shibakōen</td>
      <td>35.658068</td>
      <td>139.751599</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-16</td>
      <td>22</td>
      <td>2016-01-16</td>
      <td>Saturday</td>
      <td>0</td>
      <td>Dining bar</td>
      <td>Tōkyō-to Minato-ku Shibakōen</td>
      <td>35.658068</td>
      <td>139.751599</td>
    </tr>
    <tr>
      <th>4</th>
      <td>air_ba937bf13d40fb24</td>
      <td>2016-01-18</td>
      <td>6</td>
      <td>2016-01-18</td>
      <td>Monday</td>
      <td>0</td>
      <td>Dining bar</td>
      <td>Tōkyō-to Minato-ku Shibakōen</td>
      <td>35.658068</td>
      <td>139.751599</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Friday, Saturday & Sundays have highes visitors on avg across all stores
df_all.groupby(['day_of_week'])['visitors'].mean().to_frame().sort_values(['day_of_week']).plot()
```




    <Axes: xlabel='day_of_week'>




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_15_1.png)
    



```python
#Holidays have higher visitors on avg as compared to non holidays but its still less than friday,sat & sunday
df_all.groupby(['holiday_flg'])['visitors'].mean().to_frame().plot(kind = 'bar')
```




    <Axes: xlabel='holiday_flg'>




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_16_1.png)
    



```python
#Across all stores on avg Asian & Karoke themed restaurants have highest visitors
df_all.groupby(['air_genre_name'])['visitors'].mean().to_frame().plot(kind = 'bar')
```




    <Axes: xlabel='air_genre_name'>




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_17_1.png)
    



```python
#Top 10 areas having highest vistors on avg across all stores
df_all['air_area_name'].value_counts().sort_values(ascending = False).head(10).to_frame().plot(kind = 'bar')
```




    <Axes: xlabel='air_area_name'>




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_18_1.png)
    


Trend & Seasonality across all stores on avg visitors


```python
df_all['visit_date'] = pd.to_datetime(df_all['visit_date'])
df_all.groupby(['visit_date'])['visitors'].mean().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>visitors</th>
    </tr>
    <tr>
      <th>visit_date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01</th>
      <td>21.520833</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>29.234568</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>21.184713</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-04-18</th>
      <td>18.656985</td>
    </tr>
    <tr>
      <th>2017-04-19</th>
      <td>19.900545</td>
    </tr>
    <tr>
      <th>2017-04-20</th>
      <td>18.678238</td>
    </tr>
    <tr>
      <th>2017-04-21</th>
      <td>25.030612</td>
    </tr>
    <tr>
      <th>2017-04-22</th>
      <td>27.448320</td>
    </tr>
  </tbody>
</table>
<p>478 rows × 1 columns</p>
</div>




```python
decompose = sm.tsa.seasonal_decompose(df_all.groupby(['visit_date'])['visitors'].mean().to_frame())
figure = decompose.plot()
plt.show()
```


    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_21_0.png)
    



```python
#specificaly to observe weekly seasonality
decompose = sm.tsa.seasonal_decompose(df_all[df_all['visit_date']<='2016-01-14'].groupby(['visit_date'])['visitors'].mean().to_frame())
figure = decompose.plot()
plt.show()
```


    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_22_0.png)
    


1. There are various periods of upward & downward trends
2. Clear trend of weekly seasonality

# Modelling using stats model like ARIMA, SARIMA

Here for POC purpose we will only forecast for single store using manually configuring


```python
df_single_store = df_all[df_all['air_store_id']=='air_5c817ef28f236bdf'].reset_index(drop = True)
```


```python
df_single_store.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>air_store_id</th>
      <th>visit_date</th>
      <th>visitors</th>
      <th>calendar_date</th>
      <th>day_of_week</th>
      <th>holiday_flg</th>
      <th>air_genre_name</th>
      <th>air_area_name</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-02</td>
      <td>24</td>
      <td>2016-01-02</td>
      <td>Saturday</td>
      <td>1</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>1</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-03</td>
      <td>49</td>
      <td>2016-01-03</td>
      <td>Sunday</td>
      <td>1</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-04</td>
      <td>10</td>
      <td>2016-01-04</td>
      <td>Monday</td>
      <td>0</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-05</td>
      <td>2</td>
      <td>2016-01-05</td>
      <td>Tuesday</td>
      <td>0</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>4</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-06</td>
      <td>9</td>
      <td>2016-01-06</td>
      <td>Wednesday</td>
      <td>0</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_single_store.set_index('visit_date')['visitors'].plot()
```




    <Axes: xlabel='visit_date'>




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_28_1.png)
    



```python
#For ARIMA first we will perform ADF test to test for stationarity
adfuller(df_single_store['visitors'])
```




    (-3.9676572516950483,
     0.0015896648974107331,
     13,
     463,
     {'1%': -3.44455286264131,
      '5%': -2.8678027030003483,
      '10%': -2.5701057817594894},
     3847.930784162853)



since p-value 0.0015 < 0.05 we reject null hypothesis of series is non stationary , meaning series is stationary and doesn't require differencing


```python
#ACF & PACF Plots
sm.graphics.tsa.plot_acf(df_single_store.visitors)
plt.show()
sm.graphics.tsa.plot_pacf(df_single_store.visitors)
plt.show()
```


    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_31_0.png)
    



    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_31_1.png)
    


1. PACF plot show significant peak till 6th lag & then significant peaks at every 7th lag( indicating weekly seasonality & use of SARIMA) p = 6 , d= 0
2. ACF plot also show similar behaviour 


```python
#First trying ARIMA (before 1st apr 2017)
arima_1 = sm.tsa.ARIMA(endog = df_single_store[df_single_store['visit_date']<'2017-04-01'].visitors, order = (6,0,6)).fit()
print(arima_1.summary())
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:               visitors   No. Observations:                  455
    Model:                 ARIMA(6, 0, 6)   Log Likelihood               -1877.686
    Date:                Fri, 01 Mar 2024   AIC                           3783.371
    Time:                        10:33:19   BIC                           3841.056
    Sample:                             0   HQIC                          3806.097
                                    - 455                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         37.4330      5.998      6.241      0.000      25.677      49.189
    ar.L1          0.8068      0.103      7.847      0.000       0.605       1.008
    ar.L2         -0.4817      0.181     -2.655      0.008      -0.837      -0.126
    ar.L3          0.0319      0.228      0.140      0.888      -0.414       0.478
    ar.L4          0.3983      0.227      1.758      0.079      -0.046       0.842
    ar.L5         -0.7708      0.182     -4.238      0.000      -1.127      -0.414
    ar.L6          0.9683      0.101      9.595      0.000       0.770       1.166
    ma.L1         -0.7253      0.102     -7.122      0.000      -0.925      -0.526
    ma.L2          0.4630      0.158      2.937      0.003       0.154       0.772
    ma.L3          0.0473      0.198      0.239      0.811      -0.341       0.435
    ma.L4         -0.3415      0.186     -1.835      0.066      -0.706       0.023
    ma.L5          0.7926      0.154      5.133      0.000       0.490       1.095
    ma.L6         -0.8652      0.090     -9.628      0.000      -1.041      -0.689
    sigma2       236.7288     16.701     14.175      0.000     203.996     269.461
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                26.50
    Prob(Q):                              0.96   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.13   Skew:                             0.52
    Prob(H) (two-sided):                  0.45   Kurtosis:                         3.56
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


    /opt/conda/lib/python3.10/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "



```python
arima_1.resid.hist()
plt.show()
sm.graphics.tsa.plot_acf(arima_1.resid)
plt.show()
sm.graphics.tsa.plot_pacf(arima_1.resid)
plt.show()
```


    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_34_0.png)
    



    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_34_1.png)
    



    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_34_2.png)
    


1. p-values are for most of terms <0.05 only excluding for p = 3 & q = 3
2. residuals is close to normal distribution


```python
#trying usling SARIMA

sarima_mod = sm.tsa.statespace.SARIMAX(endog = df_single_store[df_single_store['visit_date']<'2017-04-01'].visitors,order=(6,0,6),seasonal_order = (6,0,6,7) ).fit()
print(sarima_mod.summary())
```

    /opt/conda/lib/python3.10/site-packages/statsmodels/tsa/statespace/sarimax.py:997: UserWarning: Non-stationary starting seasonal autoregressive Using zeros as starting parameters.
      warn('Non-stationary starting seasonal autoregressive'


    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =           25     M =           10
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  5.31341D+00    |proj g|=  5.33076D+01


     This problem is unconstrained.


    
    At iterate    5    f=  4.53757D+00    |proj g|=  1.63376D+00
    
    At iterate   10    f=  4.19435D+00    |proj g|=  6.31568D-01
    
    At iterate   15    f=  4.16649D+00    |proj g|=  8.35171D-02
    
    At iterate   20    f=  4.16239D+00    |proj g|=  9.12101D-02
    
    At iterate   25    f=  4.15450D+00    |proj g|=  1.47541D-01
    
    At iterate   30    f=  4.14464D+00    |proj g|=  7.57868D-02
    
    At iterate   35    f=  4.13837D+00    |proj g|=  2.65752D-02
    
    At iterate   40    f=  4.13443D+00    |proj g|=  9.01031D-02
    
    At iterate   45    f=  4.13039D+00    |proj g|=  9.30823D-02
    
    At iterate   50    f=  4.12679D+00    |proj g|=  3.16277D-02
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
       25     50     54      1     0     0   3.163D-02   4.127D+00
      F =   4.1267869650569704     
    
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 


    /opt/conda/lib/python3.10/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "


                                         SARIMAX Results                                     
    =========================================================================================
    Dep. Variable:                          visitors   No. Observations:                  455
    Model:             SARIMAX(6, 0, 6)x(6, 0, 6, 7)   Log Likelihood               -1877.688
    Date:                           Fri, 01 Mar 2024   AIC                           3805.376
    Time:                                   10:33:52   BIC                           3908.384
    Sample:                                        0   HQIC                          3845.957
                                               - 455                                         
    Covariance Type:                             opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.9338      0.141      6.641      0.000       0.658       1.209
    ar.L2         -0.5576      0.263     -2.120      0.034      -1.073      -0.042
    ar.L3          0.1061      0.327      0.325      0.745      -0.534       0.746
    ar.L4          0.3992      0.324      1.232      0.218      -0.236       1.034
    ar.L5         -0.8116      0.264     -3.076      0.002      -1.329      -0.294
    ar.L6          0.9253      0.142      6.499      0.000       0.646       1.204
    ma.L1         -0.9340      0.163     -5.746      0.000      -1.253      -0.615
    ma.L2          0.5990      0.299      2.002      0.045       0.013       1.185
    ma.L3         -0.1166      0.378     -0.309      0.758      -0.857       0.624
    ma.L4         -0.3461      0.368     -0.941      0.347      -1.067       0.375
    ma.L5          0.8107      0.297      2.727      0.006       0.228       1.394
    ma.L6         -0.8769      0.160     -5.485      0.000      -1.190      -0.564
    ar.S.L7       -0.9534      3.710     -0.257      0.797      -8.225       6.318
    ar.S.L14      -0.3151      1.225     -0.257      0.797      -2.716       2.086
    ar.S.L21       0.2757      1.138      0.242      0.809      -1.954       2.506
    ar.S.L28       0.6748      1.297      0.520      0.603      -1.867       3.217
    ar.S.L35       0.9165      1.828      0.501      0.616      -2.666       4.499
    ar.S.L42       0.3975      2.414      0.165      0.869      -4.334       5.129
    ma.S.L7        1.0508      3.696      0.284      0.776      -6.193       8.294
    ma.S.L14       0.3705      1.443      0.257      0.797      -2.458       3.199
    ma.S.L21      -0.1742      1.089     -0.160      0.873      -2.308       1.960
    ma.S.L28      -0.6346      0.997     -0.637      0.524      -2.588       1.319
    ma.S.L35      -0.9329      1.907     -0.489      0.625      -4.670       2.804
    ma.S.L42      -0.4095      2.328     -0.176      0.860      -4.972       4.153
    sigma2       261.8720     22.019     11.893      0.000     218.716     305.028
    ===================================================================================
    Ljung-Box (L1) (Q):                   6.78   Jarque-Bera (JB):                27.72
    Prob(Q):                              0.01   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.05   Skew:                             0.55
    Prob(H) (two-sided):                  0.76   Kurtosis:                         3.50
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python
sarima_mod.resid.hist()
plt.show()
sm.graphics.tsa.plot_acf(sarima_mod.resid)
plt.show()
sm.graphics.tsa.plot_pacf(sarima_mod.resid)
plt.show()
```


    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_37_0.png)
    



    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_37_1.png)
    



    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_37_2.png)
    



```python
#Calculatig MAPE values using arima
df = df_single_store[df_single_store['visit_date']>='2017-04-01'].reset_index(drop = True)['visitors'].to_frame()
df['forecast'] = pd.Series(arima_1.predict(start = 455, end= 476)).reset_index(drop = True)
df.plot(figsize=(12, 8))
```




    <Axes: >




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_38_1.png)
    



```python
((df['forecast']-df['visitors']).abs()/df['visitors']).mean()
```




    0.31286516541485976



31% MAPE using arima model


```python
#Calculatig MAPE values using sarima
df = df_single_store[df_single_store['visit_date']>='2017-04-01'].reset_index(drop = True)['visitors'].to_frame()
df['forecast'] = pd.Series(sarima_mod.predict(start = 455, end= 476)).reset_index(drop = True)
df.plot(figsize=(12, 8))
```




    <Axes: >




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_41_1.png)
    



```python
((df['forecast']-df['visitors']).abs()/df['visitors']).mean()
```




    0.3905069189832096



39% MAPE using SARIMA

# Using LSTM


```python
df_single_store.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>air_store_id</th>
      <th>visit_date</th>
      <th>visitors</th>
      <th>calendar_date</th>
      <th>day_of_week</th>
      <th>holiday_flg</th>
      <th>air_genre_name</th>
      <th>air_area_name</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-02</td>
      <td>24</td>
      <td>2016-01-02</td>
      <td>Saturday</td>
      <td>1</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>1</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-03</td>
      <td>49</td>
      <td>2016-01-03</td>
      <td>Sunday</td>
      <td>1</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-04</td>
      <td>10</td>
      <td>2016-01-04</td>
      <td>Monday</td>
      <td>0</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-05</td>
      <td>2</td>
      <td>2016-01-05</td>
      <td>Tuesday</td>
      <td>0</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
    <tr>
      <th>4</th>
      <td>air_5c817ef28f236bdf</td>
      <td>2016-01-06</td>
      <td>9</td>
      <td>2016-01-06</td>
      <td>Wednesday</td>
      <td>0</td>
      <td>Izakaya</td>
      <td>Tōkyō-to Shibuya-ku Shibuya</td>
      <td>35.661777</td>
      <td>139.704051</td>
    </tr>
  </tbody>
</table>
</div>




```python
#transforming data
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(df_single_store[df_single_store['visit_date']<'2017-04-01']['visitors'].to_frame())
```


```python
train_data_normalized.shape
```




    (455, 1)




```python
#converting to torch datatype
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
```


```python
train_data_normalized[0:5]
```




    tensor([-0.6378, -0.2441, -0.8583, -0.9843, -0.8740])



For LSTM to work we need sequence , we will pick window as 14 days to capture weekly seasonality


```python
train_window = 14
```


```python
#This functoin create a list of tuples
def create_inout_sequences(input_data, window):
    inout_seq = []
    L = len(input_data)
    for i in range(L-window):
        train_seq = input_data[i:i+window]
        train_label = input_data[i+window:i+window+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
```


```python
train_seq = create_inout_sequences(train_data_normalized, train_window)
```


```python
train_seq[0]
```




    (tensor([-0.6378, -0.2441, -0.8583, -0.9843, -0.8740, -0.7795, -0.4488, -0.3228,
             -0.6220, -0.9055, -0.9213, -1.0000, -0.9055, -0.5118]),
     tensor([-0.2441]))




```python
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out[:,-1,:])
        return predictions[-1]
```


```python
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
```


```python
model
```




    LSTM(
      (lstm): LSTM(1, 128, num_layers=2)
      (linear): Linear(in_features=128, out_features=1, bias=True)
    )




```python
#TRAINING
epochs = 100

for i in range(epochs):
    for seq, labels in train_seq:
        optimizer.zero_grad()
 
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
```

    epoch:   1 loss: 0.24456383
    epoch:  26 loss: 0.09022384
    epoch:  51 loss: 0.05866100
    epoch:  76 loss: 0.00011842
    epoch:  99 loss: 0.00010786



```python
fut_pred = 22 # size of test data from 1st apr 2017 to 22nd apr 2017

test_inputs = train_data_normalized[-train_window:].tolist() #Generating test input for first prediction
print(test_inputs)
```

    [0.10236220806837082, 0.3700787425041199, -0.4645669162273407, -0.9527559280395508, -0.4960629940032959, -0.5590550899505615, 0.08661417663097382, 0.4803149700164795, -0.4960629940032959, -0.25984251499176025, -0.6850393414497375, -0.4015747904777527, -0.4488188922405243, 0.22834645211696625]



```python
#predicting for 22 days using 14days window data
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        test_inputs.append(model(seq).item())
```


```python
test_inputs[-fut_pred:]
```




    [0.3824060261249542,
     -0.1983337700366974,
     -0.7680600881576538,
     -0.6265146732330322,
     -0.31485557556152344,
     -0.039976492524147034,
     0.14908939599990845,
     0.2971000671386719,
     -0.3498627543449402,
     -0.9400167465209961,
     -0.7354967594146729,
     -0.7206439971923828,
     0.4475521147251129,
     -0.27300992608070374,
     -0.12267790734767914,
     -0.6094772815704346,
     -0.6915880441665649,
     -0.6070899963378906,
     -0.4580090045928955,
     -0.3053012192249298,
     0.08072631061077118,
     -0.25492602586746216]




```python
#Inverse transform to get actual predictions
actual_predictions = scaler.inverse_transform(np.array(test_inputs[-fut_pred:] ).reshape(-1, 1))
print(actual_predictions)
```

    [[88.78278266]
     [51.9058056 ]
     [15.7281844 ]
     [24.71631825]
     [44.50667095]
     [61.96149272]
     [73.96717665]
     [83.36585426]
     [42.2837151 ]
     [ 4.8089366 ]
     [17.79595578]
     [18.73910618]
     [92.91955929]
     [47.16386969]
     [56.70995288]
     [25.79819262]
     [20.5841592 ]
     [25.94978523]
     [35.41642821]
     [45.11337258]
     [69.62612072]
     [48.31219736]]



```python
df['lstm_pred']  = pd.DataFrame(actual_predictions)
```


```python
#Calculatig MAPE values using lstm
df[['visitors','lstm_pred']].plot(figsize=(12, 8))
```




    <Axes: >




    
![png](https://dakshjain97.github.io/assets/img/forecasting-arima-lstm_files/forecasting-arima-lstm_64_1.png)
    



```python
((df['lstm_pred']-df['visitors']).abs()/df['visitors']).mean()
```




    0.5673198996249144



LSTM gave 56% MAPE
