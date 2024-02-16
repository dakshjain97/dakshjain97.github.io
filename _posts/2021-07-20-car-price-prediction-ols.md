---
layout: post
title: Car Price prediction using ordinary least squares
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [regression, supervised learning, OLS, linear regression assumptions]
author: Daksh Jain
---
Predict car prices of various companies (using open source kaggle dataset) by using OLS method . All steps are followed in this notebook Data collection , exploratory data analysis, feature engineering , feature selection , model building & model evaluaion . In addition to this majority of assumptions of linear regression are tested and corrected with detailed comments.

```python
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/used-car-dataset-ford-and-mercedes/cclass.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/unclean cclass.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/focus.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/toyota.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/skoda.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/ford.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/vauxhall.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/unclean focus.csv
    /kaggle/input/used-car-dataset-ford-and-mercedes/merc.csv



```python
#Importing all required libraries
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
```

# Data Collection


```python
#Combined all data
data = pd.DataFrame()
path = "/kaggle/input/used-car-dataset-ford-and-mercedes/"
for file in ['audi.csv','bmw.csv','cclass.csv','focus.csv','ford.csv','hyundi.csv','merc.csv',
            'skoda.csv','toyota.csv']:
    df = pd.read_csv(path+file)
    df['brand'] = file.split('.')[0]
    data = pd.concat([data,df])
```


```python
data.head()
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
      <th>model</th>
      <th>year</th>
      <th>price</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>brand</th>
      <th>tax(£)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A1</td>
      <td>2017</td>
      <td>12500</td>
      <td>Manual</td>
      <td>15735</td>
      <td>Petrol</td>
      <td>150.0</td>
      <td>55.4</td>
      <td>1.4</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A6</td>
      <td>2016</td>
      <td>16500</td>
      <td>Automatic</td>
      <td>36203</td>
      <td>Diesel</td>
      <td>20.0</td>
      <td>64.2</td>
      <td>2.0</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A1</td>
      <td>2016</td>
      <td>11000</td>
      <td>Manual</td>
      <td>29946</td>
      <td>Petrol</td>
      <td>30.0</td>
      <td>55.4</td>
      <td>1.4</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A4</td>
      <td>2017</td>
      <td>16800</td>
      <td>Automatic</td>
      <td>25952</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>67.3</td>
      <td>2.0</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A3</td>
      <td>2019</td>
      <td>17300</td>
      <td>Manual</td>
      <td>1998</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>49.6</td>
      <td>1.0</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (79751, 11)




```python
#describe non null values, count & data types
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 79751 entries, 0 to 6737
    Data columns (total 11 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   model         79751 non-null  object 
     1   year          79751 non-null  int64  
     2   price         79751 non-null  int64  
     3   transmission  79751 non-null  object 
     4   mileage       79751 non-null  int64  
     5   fuelType      79751 non-null  object 
     6   tax           65538 non-null  float64
     7   mpg           70398 non-null  float64
     8   engineSize    79751 non-null  float64
     9   brand         79751 non-null  object 
     10  tax(£)        4860 non-null   float64
    dtypes: float64(4), int64(3), object(4)
    memory usage: 7.3+ MB


# Data cleaning


```python
data.head()
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
      <th>model</th>
      <th>year</th>
      <th>price</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>brand</th>
      <th>tax(£)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A1</td>
      <td>2017</td>
      <td>12500</td>
      <td>Manual</td>
      <td>15735</td>
      <td>Petrol</td>
      <td>150.0</td>
      <td>55.4</td>
      <td>1.4</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A6</td>
      <td>2016</td>
      <td>16500</td>
      <td>Automatic</td>
      <td>36203</td>
      <td>Diesel</td>
      <td>20.0</td>
      <td>64.2</td>
      <td>2.0</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A1</td>
      <td>2016</td>
      <td>11000</td>
      <td>Manual</td>
      <td>29946</td>
      <td>Petrol</td>
      <td>30.0</td>
      <td>55.4</td>
      <td>1.4</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A4</td>
      <td>2017</td>
      <td>16800</td>
      <td>Automatic</td>
      <td>25952</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>67.3</td>
      <td>2.0</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A3</td>
      <td>2019</td>
      <td>17300</td>
      <td>Manual</td>
      <td>1998</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>49.6</td>
      <td>1.0</td>
      <td>audi</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#fill tax missing values and removing tax(£) column
data['tax'] = data['tax'].fillna(data['tax(£)'])
data.drop(['tax(£)'],axis = 1,inplace = True)
```


```python
#filling missing values with 0 
data.fillna(0, inplace = True)
```


```python
data.isna().sum()
```




    model           0
    year            0
    price           0
    transmission    0
    mileage         0
    fuelType        0
    tax             0
    mpg             0
    engineSize      0
    brand           0
    dtype: int64




```python
data[data.duplicated()]
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
      <th>model</th>
      <th>year</th>
      <th>price</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>brand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>273</th>
      <td>Q3</td>
      <td>2019</td>
      <td>34485</td>
      <td>Automatic</td>
      <td>10</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>47.1</td>
      <td>2.0</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>764</th>
      <td>Q2</td>
      <td>2019</td>
      <td>22495</td>
      <td>Manual</td>
      <td>1000</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>49.6</td>
      <td>1.6</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>784</th>
      <td>Q3</td>
      <td>2015</td>
      <td>13995</td>
      <td>Manual</td>
      <td>35446</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>54.3</td>
      <td>2.0</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>967</th>
      <td>Q5</td>
      <td>2019</td>
      <td>31998</td>
      <td>Semi-Auto</td>
      <td>100</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>33.2</td>
      <td>2.0</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>990</th>
      <td>Q2</td>
      <td>2019</td>
      <td>22495</td>
      <td>Manual</td>
      <td>1000</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>49.6</td>
      <td>1.6</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5489</th>
      <td>Aygo</td>
      <td>2019</td>
      <td>10350</td>
      <td>Manual</td>
      <td>2000</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>57.7</td>
      <td>1.0</td>
      <td>toyota</td>
    </tr>
    <tr>
      <th>5493</th>
      <td>Aygo</td>
      <td>2019</td>
      <td>10000</td>
      <td>Manual</td>
      <td>150</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>57.7</td>
      <td>1.0</td>
      <td>toyota</td>
    </tr>
    <tr>
      <th>5560</th>
      <td>Aygo</td>
      <td>2019</td>
      <td>10750</td>
      <td>Manual</td>
      <td>32</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>57.7</td>
      <td>1.0</td>
      <td>toyota</td>
    </tr>
    <tr>
      <th>6357</th>
      <td>Avensis</td>
      <td>2017</td>
      <td>10595</td>
      <td>Manual</td>
      <td>35939</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>67.3</td>
      <td>1.6</td>
      <td>toyota</td>
    </tr>
    <tr>
      <th>6570</th>
      <td>Hilux</td>
      <td>2015</td>
      <td>14995</td>
      <td>Automatic</td>
      <td>72100</td>
      <td>Diesel</td>
      <td>260.0</td>
      <td>32.8</td>
      <td>3.0</td>
      <td>toyota</td>
    </tr>
  </tbody>
</table>
<p>1635 rows × 10 columns</p>
</div>




```python
#removing duplicate records
data.drop_duplicates(inplace = True)
```


```python
#removing trailing and leading spaces 
data['model'] = data['model'].str.strip()
```

# EDA


```python
data.head()
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
      <th>model</th>
      <th>year</th>
      <th>price</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>brand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A1</td>
      <td>2017</td>
      <td>12500</td>
      <td>Manual</td>
      <td>15735</td>
      <td>Petrol</td>
      <td>150.0</td>
      <td>55.4</td>
      <td>1.4</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A6</td>
      <td>2016</td>
      <td>16500</td>
      <td>Automatic</td>
      <td>36203</td>
      <td>Diesel</td>
      <td>20.0</td>
      <td>64.2</td>
      <td>2.0</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A1</td>
      <td>2016</td>
      <td>11000</td>
      <td>Manual</td>
      <td>29946</td>
      <td>Petrol</td>
      <td>30.0</td>
      <td>55.4</td>
      <td>1.4</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A4</td>
      <td>2017</td>
      <td>16800</td>
      <td>Automatic</td>
      <td>25952</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>67.3</td>
      <td>2.0</td>
      <td>audi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A3</td>
      <td>2019</td>
      <td>17300</td>
      <td>Manual</td>
      <td>1998</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>49.6</td>
      <td>1.0</td>
      <td>audi</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe(percentiles = [0.05,0.25,0.50,0.75,0.95,0.99])
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
      <th>year</th>
      <th>price</th>
      <th>mileage</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>78116.000000</td>
      <td>78116.000000</td>
      <td>78116.000000</td>
      <td>78116.000000</td>
      <td>78116.000000</td>
      <td>78116.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2017.070715</td>
      <td>17984.998605</td>
      <td>23287.647998</td>
      <td>107.026025</td>
      <td>50.060259</td>
      <td>1.718827</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.152454</td>
      <td>10395.171210</td>
      <td>21352.701091</td>
      <td>71.636611</td>
      <td>24.075111</td>
      <td>0.601272</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1970.000000</td>
      <td>495.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5%</th>
      <td>2013.000000</td>
      <td>6808.500000</td>
      <td>1171.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2016.000000</td>
      <td>10890.000000</td>
      <td>7601.000000</td>
      <td>30.000000</td>
      <td>42.800000</td>
      <td>1.200000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2017.000000</td>
      <td>15857.500000</td>
      <td>17371.000000</td>
      <td>145.000000</td>
      <td>54.300000</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2019.000000</td>
      <td>22250.000000</td>
      <td>32500.000000</td>
      <td>145.000000</td>
      <td>62.800000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>95%</th>
      <td>2019.000000</td>
      <td>36249.250000</td>
      <td>66000.000000</td>
      <td>165.000000</td>
      <td>74.300000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>99%</th>
      <td>2020.000000</td>
      <td>54990.000000</td>
      <td>96306.800000</td>
      <td>269.250000</td>
      <td>85.940000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2060.000000</td>
      <td>159999.000000</td>
      <td>323000.000000</td>
      <td>580.000000</td>
      <td>470.800000</td>
      <td>6.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data['year']>2020]
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
      <th>model</th>
      <th>year</th>
      <th>price</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>brand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17726</th>
      <td>Fiesta</td>
      <td>2060</td>
      <td>6495</td>
      <td>Automatic</td>
      <td>54807</td>
      <td>Petrol</td>
      <td>205.0</td>
      <td>42.8</td>
      <td>1.4</td>
      <td>ford</td>
    </tr>
  </tbody>
</table>
</div>




```python
#removing year 2060 record could be data isse
data = data[data['year']<2060]
```


```python
#dependent variable in highly right skewed (to overcome we can transform y variable)
data['price'].hist()
```




    <Axes: >




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_20_1.png)
    



```python
#checking correlation for numerical variables with target
ls=data.select_dtypes(include=['int','float']).columns.tolist()
data[ls].corr()[['price']]

#Costly cars have lesser mileage (negative correlation)
#year & engine size has strong positive correlation with price
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
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>year</th>
      <td>0.501218</td>
    </tr>
    <tr>
      <th>price</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>-0.433683</td>
    </tr>
    <tr>
      <th>tax</th>
      <td>0.263385</td>
    </tr>
    <tr>
      <th>mpg</th>
      <td>-0.208899</td>
    </tr>
    <tr>
      <th>engineSize</th>
      <td>0.636096</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Semi - Automatic cars are most expensive
data.groupby(['transmission'])['price'].mean().plot(kind = 'bar')
```




    <Axes: xlabel='transmission'>




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_22_1.png)
    



```python
#Price by year
data.groupby(['year'])['price'].mean().plot(kind = 'bar')
```




    <Axes: xlabel='year'>




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_23_1.png)
    



```python
#considering data from 1995 to mantain data continuity across year and data cleaning
data = data[data['year']>=1995]
```


```python
#longer positive trend
data.groupby(['year'])['price'].mean().plot(kind = 'bar')
```




    <Axes: xlabel='year'>




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_25_1.png)
    



```python
#Petrol manual are cheapest and hybrid Diesel the most expensive
data.groupby(['fuelType'])['price'].mean().plot(kind = 'bar')
```




    <Axes: xlabel='fuelType'>




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_26_1.png)
    



```python
#merc has highest average price
data.groupby(['brand'])['price'].mean().plot(kind = 'bar')
```




    <Axes: xlabel='brand'>




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_27_1.png)
    



```python
#some features are left & right skewed and might need transformation for linear regression
for col in list(data.columns):
    print(col)
    data[col].hist()
    plt.show()
```

    model



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_1.png)
    


    year



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_3.png)
    


    price



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_5.png)
    


    transmission



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_7.png)
    


    mileage



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_9.png)
    


    fuelType



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_11.png)
    


    tax



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_13.png)
    


    mpg



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_15.png)
    


    engineSize



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_17.png)
    


    brand



    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_28_19.png)
    



```python
#mileage , mpg & enginesize highly right skewed
for col in ['mileage','tax','mpg','engineSize']:
    print(col)
    print(f"Skewness is {stats.skew(data[col])}")
```

    mileage
    Skewness is 1.7863557006609996
    tax
    Skewness is 0.18027160663164452
    mpg
    Skewness is 2.6096026822176115
    engineSize
    Skewness is 1.1974361566874931



```python
#Converting year to object type as some years have non linear chnage in avg prices
data['year'] = data['year'].astype(str)
```

# Feature Engineering


```python
input_feats = ['model','year','transmission','mileage','fuelType','tax','mpg','engineSize','brand']
target = ['price']
#split into train & test to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(data[input_feats], data[target], test_size=0.33, random_state=42)
```


```python
#model has many categories
data['model'].value_counts().describe(percentiles = [0.05,0.25,0.50,0.75,0.95,0.99])
```




    count     146.000000
    mean      535.020548
    std      1200.850391
    min         1.000000
    5%          1.000000
    25%        18.750000
    50%       121.500000
    75%       511.000000
    95%      1956.500000
    99%      7048.100000
    max      9313.000000
    Name: count, dtype: float64




```python
y_train.shape
```




    (52335, 1)




```python
#creating contribution of specific models price wrt other models of same brand
df_price_mean_model = pd.concat([X_train,y_train],axis = 1).groupby(['brand','model'],as_index = False)['price'].mean()
df_price_mean_brand = df_price_mean_model.groupby(['brand'],as_index = False)['price'].sum()
df_price = pd.merge(df_price_mean_model, df_price_mean_brand, on = ['brand'],how = 'inner')
del df_price_mean_model, df_price_mean_brand

df_price['price_contribution'] = round(df_price['price_x']/df_price['price_y'],3)
model_mapping = df_price[['brand','model','price_contribution']]
del df_price
```


```python
model_mapping
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
      <th>brand</th>
      <th>model</th>
      <th>price_contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>audi</td>
      <td>A1</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>audi</td>
      <td>A3</td>
      <td>0.020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>audi</td>
      <td>A4</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>3</th>
      <td>audi</td>
      <td>A5</td>
      <td>0.027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>audi</td>
      <td>A6</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141</th>
      <td>toyota</td>
      <td>Supra</td>
      <td>0.157</td>
    </tr>
    <tr>
      <th>142</th>
      <td>toyota</td>
      <td>Urban Cruiser</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>143</th>
      <td>toyota</td>
      <td>Verso</td>
      <td>0.036</td>
    </tr>
    <tr>
      <th>144</th>
      <td>toyota</td>
      <td>Verso-S</td>
      <td>0.018</td>
    </tr>
    <tr>
      <th>145</th>
      <td>toyota</td>
      <td>Yaris</td>
      <td>0.032</td>
    </tr>
  </tbody>
</table>
<p>146 rows × 3 columns</p>
</div>




```python
#combining price contribution to X_train
X_train = pd.merge(X_train,model_mapping,on = ['model','brand'],how = 'left')

#Combining price contribution to X_test and dummies to X_test
X_test = pd.merge(X_test,model_mapping,on = ['model','brand'],how = 'left')
```


```python
X_train = pd.get_dummies(X_train,columns = ['year','transmission','fuelType','brand'],drop_first = True,dtype = int)
X_train.columns
```




    Index(['model', 'mileage', 'tax', 'mpg', 'engineSize', 'price_contribution',
           'year_1997', 'year_1998', 'year_1999', 'year_2000', 'year_2001',
           'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006',
           'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
           'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016',
           'year_2017', 'year_2018', 'year_2019', 'year_2020',
           'transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto',
           'fuelType_Electric', 'fuelType_Hybrid', 'fuelType_Other',
           'fuelType_Petrol', 'brand_bmw', 'brand_cclass', 'brand_focus',
           'brand_ford', 'brand_hyundi', 'brand_merc', 'brand_skoda',
           'brand_toyota'],
          dtype='object')




```python
temp = pd.Series(X_train.columns)

#adding year dummies to test dataset
for col in list(temp[temp.str.contains('year')]):
    X_test[col] = X_test['year'].apply(lambda x: 1 if x==col.split('_')[1] else 0)
    
#adding transmission dummies to test dataset
for col in ['transmission_Manual','transmission_Other', 'transmission_Semi-Auto']:
    X_test[col] = X_test['transmission'].apply(lambda x: 1 if x==col.split('_')[1] else 0)

#adding fueltype dummies to test dataset
for col in ['fuelType_Electric','fuelType_Hybrid','fuelType_Other','fuelType_Petrol']:
    X_test[col] = X_test['fuelType'].apply(lambda x: 1 if x==col.split('_')[1] else 0)

#adding brand dummies to test dataset
for col in ['brand_bmw','brand_cclass','brand_focus','brand_ford','brand_hyundi','brand_merc','brand_skoda','brand_toyota']:
    X_test[col] = X_test['brand'].apply(lambda x: 1 if x==col.split('_')[1] else 0)
```


```python
X_test.columns
```




    Index(['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
           'engineSize', 'brand', 'price_contribution', 'year_1997', 'year_1998',
           'year_1999', 'year_2000', 'year_2001', 'year_2002', 'year_2003',
           'year_2004', 'year_2005', 'year_2006', 'year_2007', 'year_2008',
           'year_2009', 'year_2010', 'year_2011', 'year_2012', 'year_2013',
           'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018',
           'year_2019', 'year_2020', 'transmission_Manual', 'transmission_Other',
           'transmission_Semi-Auto', 'fuelType_Electric', 'fuelType_Hybrid',
           'fuelType_Other', 'fuelType_Petrol', 'brand_bmw', 'brand_cclass',
           'brand_focus', 'brand_ford', 'brand_hyundi', 'brand_merc',
           'brand_skoda', 'brand_toyota'],
          dtype='object')




```python
X_train.columns
```




    Index(['model', 'mileage', 'tax', 'mpg', 'engineSize', 'price_contribution',
           'year_1997', 'year_1998', 'year_1999', 'year_2000', 'year_2001',
           'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006',
           'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
           'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016',
           'year_2017', 'year_2018', 'year_2019', 'year_2020',
           'transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto',
           'fuelType_Electric', 'fuelType_Hybrid', 'fuelType_Other',
           'fuelType_Petrol', 'brand_bmw', 'brand_cclass', 'brand_focus',
           'brand_ford', 'brand_hyundi', 'brand_merc', 'brand_skoda',
           'brand_toyota'],
          dtype='object')




```python
X_test.head()
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
      <th>model</th>
      <th>year</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>brand</th>
      <th>price_contribution</th>
      <th>...</th>
      <th>fuelType_Other</th>
      <th>fuelType_Petrol</th>
      <th>brand_bmw</th>
      <th>brand_cclass</th>
      <th>brand_focus</th>
      <th>brand_ford</th>
      <th>brand_hyundi</th>
      <th>brand_merc</th>
      <th>brand_skoda</th>
      <th>brand_toyota</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Superb</td>
      <td>2019</td>
      <td>Semi-Auto</td>
      <td>5500</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>33.2</td>
      <td>2.0</td>
      <td>skoda</td>
      <td>0.111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Focus</td>
      <td>2005</td>
      <td>Manual</td>
      <td>87530</td>
      <td>Petrol</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>focus</td>
      <td>1.000</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Superb</td>
      <td>2019</td>
      <td>Semi-Auto</td>
      <td>7016</td>
      <td>Diesel</td>
      <td>145.0</td>
      <td>42.8</td>
      <td>2.0</td>
      <td>skoda</td>
      <td>0.111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Focus</td>
      <td>2014</td>
      <td>Manual</td>
      <td>30388</td>
      <td>Petrol</td>
      <td>235.0</td>
      <td>39.2</td>
      <td>2.0</td>
      <td>ford</td>
      <td>0.045</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>X3</td>
      <td>2019</td>
      <td>Automatic</td>
      <td>4997</td>
      <td>Petrol</td>
      <td>145.0</td>
      <td>30.4</td>
      <td>2.0</td>
      <td>bmw</td>
      <td>0.035</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



# Model Building & Feature Selection


```python
import statsmodels.api as sm
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor 
```


```python
X_train.columns
```




    Index(['model', 'mileage', 'tax', 'mpg', 'engineSize', 'price_contribution',
           'year_1997', 'year_1998', 'year_1999', 'year_2000', 'year_2001',
           'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006',
           'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
           'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016',
           'year_2017', 'year_2018', 'year_2019', 'year_2020',
           'transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto',
           'fuelType_Electric', 'fuelType_Hybrid', 'fuelType_Other',
           'fuelType_Petrol', 'brand_bmw', 'brand_cclass', 'brand_focus',
           'brand_ford', 'brand_hyundi', 'brand_merc', 'brand_skoda',
           'brand_toyota'],
          dtype='object')




```python
cols = ['mileage', 'tax', 'mpg', 'engineSize', 'year_1997',
       'year_1998', 'year_1999', 'year_2000', 'year_2001', 'year_2002',
       'year_2003', 'year_2004', 'year_2005', 'year_2006', 'year_2007',
       'year_2008', 'year_2009', 'year_2010', 'year_2011', 'year_2012',
       'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017',
       'year_2018', 'year_2019', 'year_2020', 'transmission_Manual',
       'transmission_Other', 'transmission_Semi-Auto', 'fuelType_Electric',
       'fuelType_Hybrid', 'fuelType_Other', 'fuelType_Petrol', 'brand_bmw',
       'brand_cclass', 'brand_focus', 'brand_ford', 'brand_hyundi',
       'brand_merc', 'brand_skoda', 'brand_toyota', 'price_contribution']
```


```python
#Function to fit OLS and return highest p-values variable
def fn_fit_ols(cols,y_train):
    #add constant to predictor variables
    x = sm.add_constant(X_train[cols].reset_index(drop = True))

    #fit linear regression model
    model = sm.OLS(y_train.reset_index(drop = True), x).fit()

    #view model summary
    print(model.summary())
    
    #sorting and returning highest p values
    temp = model.pvalues.sort_values(ascending = False).to_frame().reset_index().rename({0:'p-values'},axis = 1)
    temp = temp[temp['p-values']>0.05]
    
    if temp.shape[0]>0:
        return temp.sort_values(['p-values'],ascending = False)['index'][0],model
    return None,model
```


```python
#Iterating to remove step by step highest p-value variable
var,model = fn_fit_ols(cols,y_train)
while var is not None:
    print(f"After removing {var} ")
    cols.remove(var)
    var,model = fn_fit_ols(cols,y_train)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.815
    Model:                            OLS   Adj. R-squared:                  0.815
    Method:                 Least Squares   F-statistic:                     5238.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        12:37:07   Log-Likelihood:            -5.1425e+05
    No. Observations:               52335   AIC:                         1.029e+06
    Df Residuals:                   52290   BIC:                         1.029e+06
    Df Model:                          44                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                    512.2532   4486.511      0.114      0.909   -8281.351    9305.857
    mileage                   -0.0924      0.001    -62.338      0.000      -0.095      -0.089
    tax                      -15.3917      0.411    -37.421      0.000     -16.198     -14.586
    mpg                       -1.5948      1.563     -1.020      0.308      -4.658       1.468
    engineSize              8651.5050     52.743    164.031      0.000    8548.128    8754.882
    year_1997               -1.56e+04   6337.832     -2.462      0.014    -2.8e+04   -3178.981
    year_1998               -1.25e+04   4909.754     -2.545      0.011   -2.21e+04   -2873.727
    year_1999              -4798.4120   4909.198     -0.977      0.328   -1.44e+04    4823.661
    year_2000              -7132.6336   4790.738     -1.489      0.137   -1.65e+04    2257.257
    year_2001              -8707.5797   4723.970     -1.843      0.065    -1.8e+04     551.445
    year_2002              -1.144e+04   4598.016     -2.487      0.013   -2.04e+04   -2423.721
    year_2003              -6153.4685   4582.251     -1.343      0.179   -1.51e+04    2827.786
    year_2004              -5315.0740   4543.492     -1.170      0.242   -1.42e+04    3590.214
    year_2005              -4939.2990   4534.640     -1.089      0.276   -1.38e+04    3948.637
    year_2006              -5985.6155   4532.177     -1.321      0.187   -1.49e+04    2897.494
    year_2007              -3209.9092   4506.382     -0.712      0.476    -1.2e+04    5622.641
    year_2008              -2127.4370   4502.390     -0.473      0.637    -1.1e+04    6697.290
    year_2009              -1820.3370   4495.366     -0.405      0.686   -1.06e+04    6990.622
    year_2010              -1940.7325   4495.079     -0.432      0.666   -1.08e+04    6869.665
    year_2011               -985.2962   4490.814     -0.219      0.826   -9787.334    7816.742
    year_2012              -1452.9558   4487.840     -0.324      0.746   -1.02e+04    7343.253
    year_2013               -623.6430   4483.422     -0.139      0.889   -9411.192    8163.906
    year_2014                 53.3134   4483.006      0.012      0.991   -8733.420    8840.047
    year_2015                590.8746   4482.616      0.132      0.895   -8195.094    9376.843
    year_2016               1647.2346   4482.511      0.367      0.713   -7138.530    1.04e+04
    year_2017               3496.7725   4482.298      0.780      0.435   -5288.574    1.23e+04
    year_2018               5598.6747   4482.417      1.249      0.212   -3186.905    1.44e+04
    year_2019               9331.9169   4482.646      2.082      0.037     545.888    1.81e+04
    year_2020               1.342e+04   4483.788      2.993      0.003    4632.732    2.22e+04
    transmission_Manual    -1341.6024     62.737    -21.384      0.000   -1464.568   -1218.637
    transmission_Other     -3039.9655   2005.506     -1.516      0.130   -6970.776     890.845
    transmission_Semi-Auto   492.7188     58.217      8.464      0.000     378.613     606.824
    fuelType_Electric       1.143e+04   2620.258      4.363      0.000    6296.565    1.66e+04
    fuelType_Hybrid         1623.5999    130.981     12.396      0.000    1366.876    1880.324
    fuelType_Other          3342.5545    424.363      7.877      0.000    2510.800    4174.309
    fuelType_Petrol         2745.7150     52.171     52.629      0.000    2643.459    2847.971
    brand_bmw              -2301.7689     76.788    -29.976      0.000   -2452.274   -2151.264
    brand_cclass           -1.498e+05   1621.214    -92.380      0.000   -1.53e+05   -1.47e+05
    brand_focus            -1.527e+05   1640.052    -93.094      0.000   -1.56e+05   -1.49e+05
    brand_ford             -7240.7384     85.629    -84.559      0.000   -7408.573   -7072.904
    brand_hyundi           -1.395e+04    140.095    -99.555      0.000   -1.42e+04   -1.37e+04
    brand_merc             -1278.1918     74.988    -17.045      0.000   -1425.170   -1131.214
    brand_skoda            -1.418e+04    139.814   -101.432      0.000   -1.45e+04   -1.39e+04
    brand_toyota           -8358.6670    100.919    -82.826      0.000   -8556.468   -8160.866
    price_contribution      1.505e+05   1669.475     90.155      0.000    1.47e+05    1.54e+05
    ==============================================================================
    Omnibus:                    45797.184   Durbin-Watson:                   1.992
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          5876438.489
    Skew:                           3.668   Prob(JB):                         0.00
    Kurtosis:                      54.391   Cond. No.                     3.61e+07
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.61e+07. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing year_2014 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.815
    Model:                            OLS   Adj. R-squared:                  0.815
    Method:                 Least Squares   F-statistic:                     5360.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        12:37:07   Log-Likelihood:            -5.1425e+05
    No. Observations:               52335   AIC:                         1.029e+06
    Df Residuals:                   52291   BIC:                         1.029e+06
    Df Model:                          43                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                    565.5478    213.532      2.649      0.008     147.024     984.072
    mileage                   -0.0924      0.001    -62.339      0.000      -0.095      -0.089
    tax                      -15.3918      0.411    -37.426      0.000     -16.198     -14.586
    mpg                       -1.5947      1.563     -1.020      0.308      -4.658       1.468
    engineSize              8651.5027     52.742    164.034      0.000    8548.128    8754.878
    year_1997              -1.565e+04   4483.105     -3.492      0.000   -2.44e+04   -6867.540
    year_1998              -1.255e+04   2008.716     -6.248      0.000   -1.65e+04   -8613.066
    year_1999              -4851.6861   2008.096     -2.416      0.016   -8787.573    -915.799
    year_2000              -7185.9064   1698.345     -4.231      0.000   -1.05e+04   -3857.134
    year_2001              -8760.8510   1500.066     -5.840      0.000   -1.17e+04   -5820.709
    year_2002              -1.149e+04   1036.214    -11.088      0.000   -1.35e+04   -9458.161
    year_2003              -6206.7388    965.460     -6.429      0.000   -8099.050   -4314.428
    year_2004              -5368.3445    760.429     -7.060      0.000   -6858.793   -3877.896
    year_2005              -4992.5747    702.875     -7.103      0.000   -6370.215   -3614.934
    year_2006              -6038.8874    688.860     -8.766      0.000   -7389.059   -4688.716
    year_2007              -3263.1839    489.194     -6.671      0.000   -4222.009   -2304.359
    year_2008              -2180.7126    450.193     -4.844      0.000   -3063.095   -1298.330
    year_2009              -1873.6162    369.803     -5.067      0.000   -2598.434   -1148.799
    year_2010              -1994.0126    365.300     -5.459      0.000   -2710.005   -1278.020
    year_2011              -1038.5773    307.157     -3.381      0.001   -1640.609    -436.546
    year_2012              -1506.2408    254.481     -5.919      0.000   -2005.026   -1007.456
    year_2013               -676.9298    154.231     -4.389      0.000    -979.224    -374.635
    year_2015                537.5852    120.778      4.451      0.000     300.858     774.312
    year_2016               1593.9437    111.928     14.241      0.000    1374.564    1813.324
    year_2017               3443.4839    111.306     30.937      0.000    3225.324    3661.644
    year_2018               5545.3877    120.780     45.913      0.000    5308.658    5782.117
    year_2019               9278.6275    121.906     76.113      0.000    9039.690    9517.565
    year_2020               1.337e+04    156.601     85.361      0.000    1.31e+04    1.37e+04
    transmission_Manual    -1341.6041     62.737    -21.385      0.000   -1464.568   -1218.640
    transmission_Other     -3039.9682   2005.487     -1.516      0.130   -6970.741     890.804
    transmission_Semi-Auto   492.7185     58.216      8.464      0.000     378.614     606.823
    fuelType_Electric       1.143e+04   2620.232      4.363      0.000    6296.587    1.66e+04
    fuelType_Hybrid         1623.5913    130.978     12.396      0.000    1366.873    1880.309
    fuelType_Other          3342.5468    424.358      7.877      0.000    2510.801    4174.292
    fuelType_Petrol         2745.7153     52.171     52.630      0.000    2643.460    2847.970
    brand_bmw              -2301.7688     76.787    -29.976      0.000   -2452.272   -2151.265
    brand_cclass           -1.498e+05   1621.040    -92.390      0.000   -1.53e+05   -1.47e+05
    brand_focus            -1.527e+05   1639.878    -93.104      0.000   -1.56e+05   -1.49e+05
    brand_ford             -7240.7501     85.623    -84.566      0.000   -7408.572   -7072.928
    brand_hyundi           -1.395e+04    140.087    -99.560      0.000   -1.42e+04   -1.37e+04
    brand_merc             -1278.1946     74.987    -17.045      0.000   -1425.171   -1131.219
    brand_skoda            -1.418e+04    139.805   -101.439      0.000   -1.45e+04   -1.39e+04
    brand_toyota           -8358.6719    100.917    -82.827      0.000   -8556.470   -8160.874
    price_contribution      1.505e+05   1669.303     90.164      0.000    1.47e+05    1.54e+05
    ==============================================================================
    Omnibus:                    45797.179   Durbin-Watson:                   1.992
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          5876435.108
    Skew:                           3.668   Prob(JB):                         0.00
    Kurtosis:                      54.391   Cond. No.                     7.21e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.21e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing mpg 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.815
    Model:                            OLS   Adj. R-squared:                  0.815
    Method:                 Least Squares   F-statistic:                     5488.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        12:37:07   Log-Likelihood:            -5.1425e+05
    No. Observations:               52335   AIC:                         1.029e+06
    Df Residuals:                   52292   BIC:                         1.029e+06
    Df Model:                          42                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                    434.7129    170.752      2.546      0.011     100.038     769.388
    mileage                   -0.0924      0.001    -62.387      0.000      -0.095      -0.090
    tax                      -15.3166      0.405    -37.855      0.000     -16.110     -14.524
    engineSize              8667.3994     50.389    172.010      0.000    8568.637    8766.162
    year_1997              -1.565e+04   4483.107     -3.492      0.000   -2.44e+04   -6866.794
    year_1998              -1.256e+04   2008.709     -6.251      0.000   -1.65e+04   -8618.819
    year_1999              -4833.9965   2008.022     -2.407      0.016   -8769.738    -898.255
    year_2000              -7170.6989   1698.280     -4.222      0.000   -1.05e+04   -3842.053
    year_2001              -8752.7837   1500.045     -5.835      0.000   -1.17e+04   -5812.681
    year_2002              -1.149e+04   1036.214    -11.087      0.000   -1.35e+04   -9457.552
    year_2003              -6202.3401    965.451     -6.424      0.000   -8094.633   -4310.047
    year_2004              -5358.8054    760.372     -7.048      0.000   -6849.142   -3868.469
    year_2005              -4988.7919    702.865     -7.098      0.000   -6366.414   -3611.170
    year_2006              -6039.6521    688.860     -8.768      0.000   -7389.823   -4689.481
    year_2007              -3258.9757    489.177     -6.662      0.000   -4217.767   -2300.185
    year_2008              -2175.7782    450.167     -4.833      0.000   -3058.110   -1293.446
    year_2009              -1868.7045    369.772     -5.054      0.000   -2593.461   -1143.948
    year_2010              -1989.2811    365.271     -5.446      0.000   -2705.216   -1273.346
    year_2011              -1034.9442    307.137     -3.370      0.001   -1636.935    -432.953
    year_2012              -1504.4151    254.475     -5.912      0.000   -2003.188   -1005.642
    year_2013               -674.8316    154.218     -4.376      0.000    -977.100    -372.564
    year_2015                535.1741    120.755      4.432      0.000     298.492     771.856
    year_2016               1590.6808    111.882     14.217      0.000    1371.390    1809.971
    year_2017               3437.0115    111.125     30.929      0.000    3219.206    3654.817
    year_2018               5539.8606    120.658     45.914      0.000    5303.369    5776.352
    year_2019               9280.8624    121.886     76.144      0.000    9041.964    9519.761
    year_2020               1.337e+04    156.477     85.470      0.000    1.31e+04    1.37e+04
    transmission_Manual    -1346.0220     62.587    -21.506      0.000   -1468.693   -1223.351
    transmission_Other     -3032.5673   2005.474     -1.512      0.131   -6963.316     898.181
    transmission_Semi-Auto   492.3213     58.215      8.457      0.000     378.220     606.423
    fuelType_Electric       1.102e+04   2588.900      4.257      0.000    5945.749    1.61e+04
    fuelType_Hybrid         1575.5931    122.240     12.889      0.000    1336.001    1815.185
    fuelType_Other          3273.2346    418.887      7.814      0.000    2452.213    4094.256
    fuelType_Petrol         2763.0245     49.336     56.004      0.000    2666.326    2859.723
    brand_bmw              -2312.3923     76.078    -30.395      0.000   -2461.506   -2163.278
    brand_cclass           -1.499e+05   1617.126    -92.685      0.000   -1.53e+05   -1.47e+05
    brand_focus            -1.528e+05   1636.537    -93.360      0.000   -1.56e+05    -1.5e+05
    brand_ford             -7245.1955     85.512    -84.727      0.000   -7412.800   -7077.591
    brand_hyundi           -1.395e+04    140.030    -99.630      0.000   -1.42e+04   -1.37e+04
    brand_merc             -1287.1658     74.470    -17.284      0.000   -1433.128   -1141.203
    brand_skoda             -1.42e+04    139.071   -102.079      0.000   -1.45e+04   -1.39e+04
    brand_toyota           -8355.8034    100.878    -82.831      0.000   -8553.525   -8158.082
    price_contribution      1.507e+05   1656.334     90.998      0.000    1.47e+05    1.54e+05
    ==============================================================================
    Omnibus:                    45753.862   Durbin-Watson:                   1.992
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          5866308.005
    Skew:                           3.662   Prob(JB):                         0.00
    Kurtosis:                      54.347   Cond. No.                     7.21e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.21e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing transmission_Other 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.815
    Model:                            OLS   Adj. R-squared:                  0.815
    Method:                 Least Squares   F-statistic:                     5621.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        12:37:07   Log-Likelihood:            -5.1425e+05
    No. Observations:               52335   AIC:                         1.029e+06
    Df Residuals:                   52293   BIC:                         1.029e+06
    Df Model:                          41                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                    433.6328    170.752      2.540      0.011      98.957     768.309
    mileage                   -0.0924      0.001    -62.378      0.000      -0.095      -0.090
    tax                      -15.3141      0.405    -37.849      0.000     -16.107     -14.521
    engineSize              8667.2644     50.389    172.005      0.000    8568.501    8766.028
    year_1997              -1.565e+04   4483.162     -3.492      0.000   -2.44e+04   -6866.621
    year_1998              -1.256e+04   2008.734     -6.251      0.000   -1.65e+04   -8619.137
    year_1999              -4834.9716   2008.046     -2.408      0.016   -8770.761    -899.182
    year_2000              -7170.8561   1698.301     -4.222      0.000   -1.05e+04   -3842.169
    year_2001              -8753.1088   1500.064     -5.835      0.000   -1.17e+04   -5812.970
    year_2002              -1.149e+04   1036.227    -11.087      0.000   -1.35e+04   -9457.741
    year_2003              -6203.0649    965.463     -6.425      0.000   -8095.381   -4310.749
    year_2004              -5359.8423    760.381     -7.049      0.000   -6850.196   -3869.488
    year_2005              -4989.7895    702.873     -7.099      0.000   -6367.428   -3612.151
    year_2006              -6040.6555    688.868     -8.769      0.000   -7390.843   -4690.468
    year_2007              -3259.8956    489.182     -6.664      0.000   -4218.698   -2301.093
    year_2008              -2176.8145    450.172     -4.836      0.000   -3059.157   -1294.472
    year_2009              -1869.4628    369.776     -5.056      0.000   -2594.228   -1144.698
    year_2010              -1989.8943    365.275     -5.448      0.000   -2705.838   -1273.951
    year_2011              -1035.4103    307.140     -3.371      0.001   -1637.409    -433.412
    year_2012              -1512.9432    254.415     -5.947      0.000   -2011.600   -1014.287
    year_2013               -674.9862    154.219     -4.377      0.000    -977.258    -372.715
    year_2015                533.7561    120.753      4.420      0.000     297.079     770.434
    year_2016               1590.9422    111.884     14.220      0.000    1371.649    1810.235
    year_2017               3436.7497    111.126     30.927      0.000    3218.942    3654.558
    year_2018               5540.1591    120.659     45.916      0.000    5303.665    5776.653
    year_2019               9281.4066    121.887     76.147      0.000    9042.506    9520.307
    year_2020               1.337e+04    156.478     85.473      0.000    1.31e+04    1.37e+04
    transmission_Manual    -1344.9627     62.584    -21.491      0.000   -1467.628   -1222.298
    transmission_Semi-Auto   493.1756     58.213      8.472      0.000     379.078     607.273
    fuelType_Electric       1.102e+04   2588.932      4.257      0.000    5946.456    1.61e+04
    fuelType_Hybrid         1573.5128    122.234     12.873      0.000    1333.933    1813.093
    fuelType_Other          3273.8084    418.892      7.815      0.000    2452.777    4094.840
    fuelType_Petrol         2762.9467     49.337     56.002      0.000    2666.247    2859.647
    brand_bmw              -2312.2550     76.079    -30.393      0.000   -2461.371   -2163.139
    brand_cclass           -1.499e+05   1617.146    -92.684      0.000   -1.53e+05   -1.47e+05
    brand_focus            -1.528e+05   1636.557    -93.358      0.000   -1.56e+05    -1.5e+05
    brand_ford             -7245.3889     85.513    -84.728      0.000   -7412.995   -7077.782
    brand_hyundi           -1.395e+04    140.026    -99.647      0.000   -1.42e+04   -1.37e+04
    brand_merc             -1287.3629     74.471    -17.287      0.000   -1433.327   -1141.399
    brand_skoda             -1.42e+04    139.073   -102.079      0.000   -1.45e+04   -1.39e+04
    brand_toyota           -8355.6468    100.879    -82.829      0.000   -8553.370   -8157.923
    price_contribution      1.507e+05   1656.354     90.997      0.000    1.47e+05    1.54e+05
    ==============================================================================
    Omnibus:                    45752.307   Durbin-Watson:                   1.992
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          5865715.137
    Skew:                           3.662   Prob(JB):                         0.00
    Kurtosis:                      54.345   Cond. No.                     7.21e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.21e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#final model
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.815</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.815</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   5621.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 16 Feb 2024</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:37:07</td>     <th>  Log-Likelihood:    </th> <td>-5.1425e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 52335</td>      <th>  AIC:               </th>  <td>1.029e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 52293</td>      <th>  BIC:               </th>  <td>1.029e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    41</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                  <td>  433.6328</td> <td>  170.752</td> <td>    2.540</td> <td> 0.011</td> <td>   98.957</td> <td>  768.309</td>
</tr>
<tr>
  <th>mileage</th>                <td>   -0.0924</td> <td>    0.001</td> <td>  -62.378</td> <td> 0.000</td> <td>   -0.095</td> <td>   -0.090</td>
</tr>
<tr>
  <th>tax</th>                    <td>  -15.3141</td> <td>    0.405</td> <td>  -37.849</td> <td> 0.000</td> <td>  -16.107</td> <td>  -14.521</td>
</tr>
<tr>
  <th>engineSize</th>             <td> 8667.2644</td> <td>   50.389</td> <td>  172.005</td> <td> 0.000</td> <td> 8568.501</td> <td> 8766.028</td>
</tr>
<tr>
  <th>year_1997</th>              <td>-1.565e+04</td> <td> 4483.162</td> <td>   -3.492</td> <td> 0.000</td> <td>-2.44e+04</td> <td>-6866.621</td>
</tr>
<tr>
  <th>year_1998</th>              <td>-1.256e+04</td> <td> 2008.734</td> <td>   -6.251</td> <td> 0.000</td> <td>-1.65e+04</td> <td>-8619.137</td>
</tr>
<tr>
  <th>year_1999</th>              <td>-4834.9716</td> <td> 2008.046</td> <td>   -2.408</td> <td> 0.016</td> <td>-8770.761</td> <td> -899.182</td>
</tr>
<tr>
  <th>year_2000</th>              <td>-7170.8561</td> <td> 1698.301</td> <td>   -4.222</td> <td> 0.000</td> <td>-1.05e+04</td> <td>-3842.169</td>
</tr>
<tr>
  <th>year_2001</th>              <td>-8753.1088</td> <td> 1500.064</td> <td>   -5.835</td> <td> 0.000</td> <td>-1.17e+04</td> <td>-5812.970</td>
</tr>
<tr>
  <th>year_2002</th>              <td>-1.149e+04</td> <td> 1036.227</td> <td>  -11.087</td> <td> 0.000</td> <td>-1.35e+04</td> <td>-9457.741</td>
</tr>
<tr>
  <th>year_2003</th>              <td>-6203.0649</td> <td>  965.463</td> <td>   -6.425</td> <td> 0.000</td> <td>-8095.381</td> <td>-4310.749</td>
</tr>
<tr>
  <th>year_2004</th>              <td>-5359.8423</td> <td>  760.381</td> <td>   -7.049</td> <td> 0.000</td> <td>-6850.196</td> <td>-3869.488</td>
</tr>
<tr>
  <th>year_2005</th>              <td>-4989.7895</td> <td>  702.873</td> <td>   -7.099</td> <td> 0.000</td> <td>-6367.428</td> <td>-3612.151</td>
</tr>
<tr>
  <th>year_2006</th>              <td>-6040.6555</td> <td>  688.868</td> <td>   -8.769</td> <td> 0.000</td> <td>-7390.843</td> <td>-4690.468</td>
</tr>
<tr>
  <th>year_2007</th>              <td>-3259.8956</td> <td>  489.182</td> <td>   -6.664</td> <td> 0.000</td> <td>-4218.698</td> <td>-2301.093</td>
</tr>
<tr>
  <th>year_2008</th>              <td>-2176.8145</td> <td>  450.172</td> <td>   -4.836</td> <td> 0.000</td> <td>-3059.157</td> <td>-1294.472</td>
</tr>
<tr>
  <th>year_2009</th>              <td>-1869.4628</td> <td>  369.776</td> <td>   -5.056</td> <td> 0.000</td> <td>-2594.228</td> <td>-1144.698</td>
</tr>
<tr>
  <th>year_2010</th>              <td>-1989.8943</td> <td>  365.275</td> <td>   -5.448</td> <td> 0.000</td> <td>-2705.838</td> <td>-1273.951</td>
</tr>
<tr>
  <th>year_2011</th>              <td>-1035.4103</td> <td>  307.140</td> <td>   -3.371</td> <td> 0.001</td> <td>-1637.409</td> <td> -433.412</td>
</tr>
<tr>
  <th>year_2012</th>              <td>-1512.9432</td> <td>  254.415</td> <td>   -5.947</td> <td> 0.000</td> <td>-2011.600</td> <td>-1014.287</td>
</tr>
<tr>
  <th>year_2013</th>              <td> -674.9862</td> <td>  154.219</td> <td>   -4.377</td> <td> 0.000</td> <td> -977.258</td> <td> -372.715</td>
</tr>
<tr>
  <th>year_2015</th>              <td>  533.7561</td> <td>  120.753</td> <td>    4.420</td> <td> 0.000</td> <td>  297.079</td> <td>  770.434</td>
</tr>
<tr>
  <th>year_2016</th>              <td> 1590.9422</td> <td>  111.884</td> <td>   14.220</td> <td> 0.000</td> <td> 1371.649</td> <td> 1810.235</td>
</tr>
<tr>
  <th>year_2017</th>              <td> 3436.7497</td> <td>  111.126</td> <td>   30.927</td> <td> 0.000</td> <td> 3218.942</td> <td> 3654.558</td>
</tr>
<tr>
  <th>year_2018</th>              <td> 5540.1591</td> <td>  120.659</td> <td>   45.916</td> <td> 0.000</td> <td> 5303.665</td> <td> 5776.653</td>
</tr>
<tr>
  <th>year_2019</th>              <td> 9281.4066</td> <td>  121.887</td> <td>   76.147</td> <td> 0.000</td> <td> 9042.506</td> <td> 9520.307</td>
</tr>
<tr>
  <th>year_2020</th>              <td> 1.337e+04</td> <td>  156.478</td> <td>   85.473</td> <td> 0.000</td> <td> 1.31e+04</td> <td> 1.37e+04</td>
</tr>
<tr>
  <th>transmission_Manual</th>    <td>-1344.9627</td> <td>   62.584</td> <td>  -21.491</td> <td> 0.000</td> <td>-1467.628</td> <td>-1222.298</td>
</tr>
<tr>
  <th>transmission_Semi-Auto</th> <td>  493.1756</td> <td>   58.213</td> <td>    8.472</td> <td> 0.000</td> <td>  379.078</td> <td>  607.273</td>
</tr>
<tr>
  <th>fuelType_Electric</th>      <td> 1.102e+04</td> <td> 2588.932</td> <td>    4.257</td> <td> 0.000</td> <td> 5946.456</td> <td> 1.61e+04</td>
</tr>
<tr>
  <th>fuelType_Hybrid</th>        <td> 1573.5128</td> <td>  122.234</td> <td>   12.873</td> <td> 0.000</td> <td> 1333.933</td> <td> 1813.093</td>
</tr>
<tr>
  <th>fuelType_Other</th>         <td> 3273.8084</td> <td>  418.892</td> <td>    7.815</td> <td> 0.000</td> <td> 2452.777</td> <td> 4094.840</td>
</tr>
<tr>
  <th>fuelType_Petrol</th>        <td> 2762.9467</td> <td>   49.337</td> <td>   56.002</td> <td> 0.000</td> <td> 2666.247</td> <td> 2859.647</td>
</tr>
<tr>
  <th>brand_bmw</th>              <td>-2312.2550</td> <td>   76.079</td> <td>  -30.393</td> <td> 0.000</td> <td>-2461.371</td> <td>-2163.139</td>
</tr>
<tr>
  <th>brand_cclass</th>           <td>-1.499e+05</td> <td> 1617.146</td> <td>  -92.684</td> <td> 0.000</td> <td>-1.53e+05</td> <td>-1.47e+05</td>
</tr>
<tr>
  <th>brand_focus</th>            <td>-1.528e+05</td> <td> 1636.557</td> <td>  -93.358</td> <td> 0.000</td> <td>-1.56e+05</td> <td> -1.5e+05</td>
</tr>
<tr>
  <th>brand_ford</th>             <td>-7245.3889</td> <td>   85.513</td> <td>  -84.728</td> <td> 0.000</td> <td>-7412.995</td> <td>-7077.782</td>
</tr>
<tr>
  <th>brand_hyundi</th>           <td>-1.395e+04</td> <td>  140.026</td> <td>  -99.647</td> <td> 0.000</td> <td>-1.42e+04</td> <td>-1.37e+04</td>
</tr>
<tr>
  <th>brand_merc</th>             <td>-1287.3629</td> <td>   74.471</td> <td>  -17.287</td> <td> 0.000</td> <td>-1433.327</td> <td>-1141.399</td>
</tr>
<tr>
  <th>brand_skoda</th>            <td> -1.42e+04</td> <td>  139.073</td> <td> -102.079</td> <td> 0.000</td> <td>-1.45e+04</td> <td>-1.39e+04</td>
</tr>
<tr>
  <th>brand_toyota</th>           <td>-8355.6468</td> <td>  100.879</td> <td>  -82.829</td> <td> 0.000</td> <td>-8553.370</td> <td>-8157.923</td>
</tr>
<tr>
  <th>price_contribution</th>     <td> 1.507e+05</td> <td> 1656.354</td> <td>   90.997</td> <td> 0.000</td> <td> 1.47e+05</td> <td> 1.54e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>45752.307</td> <th>  Durbin-Watson:     </th>  <td>   1.992</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>5865715.137</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.662</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>54.345</td>   <th>  Cond. No.          </th>  <td>7.21e+06</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 7.21e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
#test r-square is also 0.81
X_test_ols = sm.add_constant(X_test[cols])
r2_score(y_test.reset_index(drop = True)['price'],model.predict(X_test_ols).fillna(0))
```




    0.8134458898305428




```python
#model coefficients
model.params
```




    const                        433.632793
    mileage                       -0.092421
    tax                          -15.314096
    engineSize                  8667.264390
    year_1997                 -15653.659756
    year_1998                 -12556.274476
    year_1999                  -4834.971578
    year_2000                  -7170.856066
    year_2001                  -8753.108812
    year_2002                 -11488.755083
    year_2003                  -6203.064878
    year_2004                  -5359.842307
    year_2005                  -4989.789483
    year_2006                  -6040.655459
    year_2007                  -3259.895629
    year_2008                  -2176.814466
    year_2009                  -1869.462756
    year_2010                  -1989.894313
    year_2011                  -1035.410336
    year_2012                  -1512.943182
    year_2013                   -674.986224
    year_2015                    533.756050
    year_2016                   1590.942235
    year_2017                   3436.749685
    year_2018                   5540.159052
    year_2019                   9281.406560
    year_2020                  13374.634556
    transmission_Manual        -1344.962700
    transmission_Semi-Auto       493.175627
    fuelType_Electric          11020.786832
    fuelType_Hybrid             1573.512804
    fuelType_Other              3273.808359
    fuelType_Petrol             2762.946730
    brand_bmw                  -2312.255047
    brand_cclass             -149883.166026
    brand_focus              -152785.545594
    brand_ford                 -7245.388859
    brand_hyundi              -13953.116582
    brand_merc                 -1287.362918
    brand_skoda               -14196.374115
    brand_toyota               -8355.646783
    price_contribution        150723.020232
    dtype: float64




```python
#checking OLS method assumptions

# 1. No autocorrelation of residuals -> Durbin-Watson value <2 suggests there is no autocorrelation of residuals

# 2. Homoscedasticity -> by plotting residuals vs fitted values
plt.scatter(model.fittedvalues, model.fittedvalues - y_train.reset_index()['price'])
plt.show()

#we can clearly notice as x increase the residuals also increases meaning it is violating homoscedaticity assumption
#to overcome this we can transform our y variable then fit it 
```


    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_52_0.png)
    



```python
#3. Another assumption is residuals should be normally distrbuted which is not the case here its highly left skewed 
(model.fittedvalues - y_train.reset_index()['price']).hist()
```




    <Axes: >




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_53_1.png)
    



```python
#Resetting columns to all variables
cols = ['mileage', 'tax', 'mpg', 'engineSize', 'year_1997',
       'year_1998', 'year_1999', 'year_2000', 'year_2001', 'year_2002',
       'year_2003', 'year_2004', 'year_2005', 'year_2006', 'year_2007',
       'year_2008', 'year_2009', 'year_2010', 'year_2011', 'year_2012',
       'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017',
       'year_2018', 'year_2019', 'year_2020', 'transmission_Manual',
       'transmission_Other', 'transmission_Semi-Auto', 'fuelType_Electric',
       'fuelType_Hybrid', 'fuelType_Other', 'fuelType_Petrol', 'brand_bmw',
       'brand_cclass', 'brand_focus', 'brand_ford', 'brand_hyundi',
       'brand_merc', 'brand_skoda', 'brand_toyota', 'price_contribution']
```


```python
#4. Now we can check for multicolinearity assumption by checkin VIF values and
vif_data = pd.DataFrame() 
vif_data["feature"] = cols
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X_train[cols].values, i) 
                          for i in range(len(X_train[cols].columns))]
```


```python
#Here we can see some features like price_contribution,brand_focus,brand_cclass,engineSize,year_2019,mpg etc have very high VIF value we can try removing it step by step
vif_data.sort_values(['VIF'],ascending = False)
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
      <th>feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43</th>
      <td>price_contribution</td>
      <td>813.421442</td>
    </tr>
    <tr>
      <th>37</th>
      <td>brand_focus</td>
      <td>425.552842</td>
    </tr>
    <tr>
      <th>36</th>
      <td>brand_cclass</td>
      <td>336.908676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>engineSize</td>
      <td>24.053660</td>
    </tr>
    <tr>
      <th>26</th>
      <td>year_2019</td>
      <td>22.761359</td>
    </tr>
    <tr>
      <th>24</th>
      <td>year_2017</td>
      <td>21.127788</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mpg</td>
      <td>19.820582</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year_2016</td>
      <td>14.345850</td>
    </tr>
    <tr>
      <th>25</th>
      <td>year_2018</td>
      <td>13.952727</td>
    </tr>
    <tr>
      <th>22</th>
      <td>year_2015</td>
      <td>8.117300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tax</td>
      <td>7.302212</td>
    </tr>
    <tr>
      <th>0</th>
      <td>mileage</td>
      <td>5.684236</td>
    </tr>
    <tr>
      <th>28</th>
      <td>transmission_Manual</td>
      <td>5.118133</td>
    </tr>
    <tr>
      <th>21</th>
      <td>year_2014</td>
      <td>4.775303</td>
    </tr>
    <tr>
      <th>38</th>
      <td>brand_ford</td>
      <td>4.326524</td>
    </tr>
    <tr>
      <th>41</th>
      <td>brand_skoda</td>
      <td>4.057231</td>
    </tr>
    <tr>
      <th>27</th>
      <td>year_2020</td>
      <td>4.021835</td>
    </tr>
    <tr>
      <th>20</th>
      <td>year_2013</td>
      <td>3.620132</td>
    </tr>
    <tr>
      <th>34</th>
      <td>fuelType_Petrol</td>
      <td>3.588385</td>
    </tr>
    <tr>
      <th>39</th>
      <td>brand_hyundi</td>
      <td>3.165565</td>
    </tr>
    <tr>
      <th>40</th>
      <td>brand_merc</td>
      <td>2.391344</td>
    </tr>
    <tr>
      <th>30</th>
      <td>transmission_Semi-Auto</td>
      <td>2.296453</td>
    </tr>
    <tr>
      <th>42</th>
      <td>brand_toyota</td>
      <td>2.286775</td>
    </tr>
    <tr>
      <th>35</th>
      <td>brand_bmw</td>
      <td>2.107131</td>
    </tr>
    <tr>
      <th>32</th>
      <td>fuelType_Hybrid</td>
      <td>1.773921</td>
    </tr>
    <tr>
      <th>19</th>
      <td>year_2012</td>
      <td>1.744541</td>
    </tr>
    <tr>
      <th>18</th>
      <td>year_2011</td>
      <td>1.536992</td>
    </tr>
    <tr>
      <th>17</th>
      <td>year_2010</td>
      <td>1.401281</td>
    </tr>
    <tr>
      <th>16</th>
      <td>year_2009</td>
      <td>1.400448</td>
    </tr>
    <tr>
      <th>15</th>
      <td>year_2008</td>
      <td>1.302394</td>
    </tr>
    <tr>
      <th>14</th>
      <td>year_2007</td>
      <td>1.242055</td>
    </tr>
    <tr>
      <th>13</th>
      <td>year_2006</td>
      <td>1.130225</td>
    </tr>
    <tr>
      <th>12</th>
      <td>year_2005</td>
      <td>1.115006</td>
    </tr>
    <tr>
      <th>11</th>
      <td>year_2004</td>
      <td>1.110752</td>
    </tr>
    <tr>
      <th>10</th>
      <td>year_2003</td>
      <td>1.065943</td>
    </tr>
    <tr>
      <th>33</th>
      <td>fuelType_Other</td>
      <td>1.058387</td>
    </tr>
    <tr>
      <th>9</th>
      <td>year_2002</td>
      <td>1.052916</td>
    </tr>
    <tr>
      <th>31</th>
      <td>fuelType_Electric</td>
      <td>1.025877</td>
    </tr>
    <tr>
      <th>8</th>
      <td>year_2001</td>
      <td>1.024779</td>
    </tr>
    <tr>
      <th>5</th>
      <td>year_1998</td>
      <td>1.017411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>year_2000</td>
      <td>1.017140</td>
    </tr>
    <tr>
      <th>6</th>
      <td>year_1999</td>
      <td>1.012191</td>
    </tr>
    <tr>
      <th>4</th>
      <td>year_1997</td>
      <td>1.003285</td>
    </tr>
    <tr>
      <th>29</th>
      <td>transmission_Other</td>
      <td>1.001628</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking multicollinearity after removing brand_focus(removing second larget VIF feature as price_controbution is a feature we calculated)
cols.remove('brand_focus')
vif_data = pd.DataFrame() 
vif_data["feature"] = cols

# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X_train[cols].values, i) 
                          for i in range(len(X_train[cols].columns))]
```


```python
#Here we can observe just by removing brand_focus variable major multicolinearity is solved, here we can now fit 
vif_data.sort_values(['VIF'],ascending = False)
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
      <th>feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>year_2019</td>
      <td>22.500103</td>
    </tr>
    <tr>
      <th>3</th>
      <td>engineSize</td>
      <td>21.546025</td>
    </tr>
    <tr>
      <th>24</th>
      <td>year_2017</td>
      <td>21.038207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mpg</td>
      <td>19.738932</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year_2016</td>
      <td>14.287094</td>
    </tr>
    <tr>
      <th>25</th>
      <td>year_2018</td>
      <td>13.864800</td>
    </tr>
    <tr>
      <th>22</th>
      <td>year_2015</td>
      <td>8.101215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tax</td>
      <td>7.192179</td>
    </tr>
    <tr>
      <th>42</th>
      <td>price_contribution</td>
      <td>6.576551</td>
    </tr>
    <tr>
      <th>0</th>
      <td>mileage</td>
      <td>5.674358</td>
    </tr>
    <tr>
      <th>28</th>
      <td>transmission_Manual</td>
      <td>5.057088</td>
    </tr>
    <tr>
      <th>21</th>
      <td>year_2014</td>
      <td>4.770928</td>
    </tr>
    <tr>
      <th>27</th>
      <td>year_2020</td>
      <td>3.971462</td>
    </tr>
    <tr>
      <th>20</th>
      <td>year_2013</td>
      <td>3.617872</td>
    </tr>
    <tr>
      <th>34</th>
      <td>fuelType_Petrol</td>
      <td>3.478167</td>
    </tr>
    <tr>
      <th>37</th>
      <td>brand_ford</td>
      <td>3.143197</td>
    </tr>
    <tr>
      <th>39</th>
      <td>brand_merc</td>
      <td>2.342264</td>
    </tr>
    <tr>
      <th>30</th>
      <td>transmission_Semi-Auto</td>
      <td>2.293545</td>
    </tr>
    <tr>
      <th>35</th>
      <td>brand_bmw</td>
      <td>2.106372</td>
    </tr>
    <tr>
      <th>36</th>
      <td>brand_cclass</td>
      <td>2.064793</td>
    </tr>
    <tr>
      <th>41</th>
      <td>brand_toyota</td>
      <td>2.063635</td>
    </tr>
    <tr>
      <th>32</th>
      <td>fuelType_Hybrid</td>
      <td>1.762246</td>
    </tr>
    <tr>
      <th>19</th>
      <td>year_2012</td>
      <td>1.744055</td>
    </tr>
    <tr>
      <th>40</th>
      <td>brand_skoda</td>
      <td>1.613828</td>
    </tr>
    <tr>
      <th>18</th>
      <td>year_2011</td>
      <td>1.536959</td>
    </tr>
    <tr>
      <th>38</th>
      <td>brand_hyundi</td>
      <td>1.524370</td>
    </tr>
    <tr>
      <th>17</th>
      <td>year_2010</td>
      <td>1.401278</td>
    </tr>
    <tr>
      <th>16</th>
      <td>year_2009</td>
      <td>1.400447</td>
    </tr>
    <tr>
      <th>15</th>
      <td>year_2008</td>
      <td>1.302316</td>
    </tr>
    <tr>
      <th>14</th>
      <td>year_2007</td>
      <td>1.241978</td>
    </tr>
    <tr>
      <th>13</th>
      <td>year_2006</td>
      <td>1.130157</td>
    </tr>
    <tr>
      <th>12</th>
      <td>year_2005</td>
      <td>1.115002</td>
    </tr>
    <tr>
      <th>11</th>
      <td>year_2004</td>
      <td>1.110532</td>
    </tr>
    <tr>
      <th>10</th>
      <td>year_2003</td>
      <td>1.065444</td>
    </tr>
    <tr>
      <th>33</th>
      <td>fuelType_Other</td>
      <td>1.058355</td>
    </tr>
    <tr>
      <th>9</th>
      <td>year_2002</td>
      <td>1.052914</td>
    </tr>
    <tr>
      <th>31</th>
      <td>fuelType_Electric</td>
      <td>1.025616</td>
    </tr>
    <tr>
      <th>8</th>
      <td>year_2001</td>
      <td>1.024729</td>
    </tr>
    <tr>
      <th>5</th>
      <td>year_1998</td>
      <td>1.017390</td>
    </tr>
    <tr>
      <th>7</th>
      <td>year_2000</td>
      <td>1.016876</td>
    </tr>
    <tr>
      <th>6</th>
      <td>year_1999</td>
      <td>1.012143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>year_1997</td>
      <td>1.003284</td>
    </tr>
    <tr>
      <th>29</th>
      <td>transmission_Other</td>
      <td>1.001628</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Performing log transformation on target and fitting again after removing brand_focus variable
#Iterating to remove step by step highest p-value variable
var,model = fn_fit_ols(cols,np.log(y_train))
while var is not None:
    print(f"After removing {var} ")
    cols.remove(var)
    var,model = fn_fit_ols(cols,np.log(y_train))
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.882
    Model:                            OLS   Adj. R-squared:                  0.882
    Method:                 Least Squares   F-statistic:                     9064.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:43:30   Log-Likelihood:                 14242.
    No. Observations:               52335   AIC:                        -2.840e+04
    Df Residuals:                   52291   BIC:                        -2.801e+04
    Df Model:                          43                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                      7.9486      0.185     43.054      0.000       7.587       8.311
    mileage                -5.057e-06   6.09e-08    -82.976      0.000   -5.18e-06   -4.94e-06
    tax                        0.0005   1.68e-05     28.772      0.000       0.000       0.001
    mpg                       -0.0010   6.42e-05    -16.165      0.000      -0.001      -0.001
    engineSize                 0.3544      0.002    172.507      0.000       0.350       0.358
    year_1997                  0.4784      0.261      1.834      0.067      -0.033       0.990
    year_1998                  0.0430      0.202      0.213      0.831      -0.353       0.439
    year_1999                 -0.1677      0.202     -0.830      0.407      -0.564       0.228
    year_2000                 -0.5729      0.197     -2.906      0.004      -0.959      -0.186
    year_2001                 -0.0141      0.194     -0.072      0.942      -0.395       0.367
    year_2002                 -0.5034      0.189     -2.661      0.008      -0.874      -0.133
    year_2003                 -0.2060      0.189     -1.092      0.275      -0.576       0.164
    year_2004                  0.0234      0.187      0.125      0.900      -0.343       0.390
    year_2005                 -0.1489      0.187     -0.798      0.425      -0.515       0.217
    year_2006                 -0.0202      0.187     -0.108      0.914      -0.386       0.345
    year_2007                  0.1188      0.185      0.640      0.522      -0.245       0.482
    year_2008                  0.2564      0.185      1.384      0.166      -0.107       0.620
    year_2009                  0.4364      0.185      2.359      0.018       0.074       0.799
    year_2010                  0.5773      0.185      3.121      0.002       0.215       0.940
    year_2011                  0.7600      0.185      4.113      0.000       0.398       1.122
    year_2012                  0.8998      0.185      4.873      0.000       0.538       1.262
    year_2013                  1.0207      0.184      5.533      0.000       0.659       1.382
    year_2014                  1.1414      0.184      6.187      0.000       0.780       1.503
    year_2015                  1.2451      0.184      6.750      0.000       0.884       1.607
    year_2016                  1.3616      0.184      7.382      0.000       1.000       1.723
    year_2017                  1.4346      0.184      7.778      0.000       1.073       1.796
    year_2018                  1.5366      0.184      8.331      0.000       1.175       1.898
    year_2019                  1.7051      0.184      9.244      0.000       1.344       2.067
    year_2020                  1.8290      0.184      9.914      0.000       1.467       2.191
    transmission_Manual       -0.1395      0.003    -54.369      0.000      -0.145      -0.135
    transmission_Other        -0.0597      0.083     -0.724      0.469      -0.222       0.102
    transmission_Semi-Auto     0.0006      0.002      0.236      0.813      -0.004       0.005
    fuelType_Electric          0.7816      0.108      7.250      0.000       0.570       0.993
    fuelType_Hybrid            0.2548      0.005     47.418      0.000       0.244       0.265
    fuelType_Other             0.1584      0.017      9.068      0.000       0.124       0.193
    fuelType_Petrol           -0.0239      0.002    -11.309      0.000      -0.028      -0.020
    brand_bmw                 -0.1106      0.003    -34.999      0.000      -0.117      -0.104
    brand_cclass               0.1309      0.005     25.071      0.000       0.121       0.141
    brand_ford                -0.2543      0.003    -84.687      0.000      -0.260      -0.248
    brand_hyundi              -0.3611      0.004    -90.264      0.000      -0.369      -0.353
    brand_merc                -0.0283      0.003     -9.264      0.000      -0.034      -0.022
    brand_skoda               -0.2866      0.004    -78.990      0.000      -0.294      -0.280
    brand_toyota              -0.4130      0.004   -104.685      0.000      -0.421      -0.405
    price_contribution        -0.1876      0.006    -30.360      0.000      -0.200      -0.176
    ==============================================================================
    Omnibus:                     4316.050   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20483.041
    Skew:                           0.268   Prob(JB):                         0.00
    Kurtosis:                       6.017   Cond. No.                     3.61e+07
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.61e+07. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing year_2001 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.882
    Model:                            OLS   Adj. R-squared:                  0.882
    Method:                 Least Squares   F-statistic:                     9280.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:43:30   Log-Likelihood:                 14242.
    No. Observations:               52335   AIC:                        -2.840e+04
    Df Residuals:                   52292   BIC:                        -2.802e+04
    Df Model:                          42                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                      7.9360      0.059    134.300      0.000       7.820       8.052
    mileage                -5.057e-06   6.09e-08    -82.979      0.000   -5.18e-06   -4.94e-06
    tax                        0.0005   1.68e-05     28.772      0.000       0.000       0.001
    mpg                       -0.0010   6.42e-05    -16.165      0.000      -0.001      -0.001
    engineSize                 0.3544      0.002    172.508      0.000       0.350       0.358
    year_1997                  0.4911      0.193      2.539      0.011       0.112       0.870
    year_1998                  0.0557      0.101      0.551      0.582      -0.142       0.254
    year_1999                 -0.1550      0.101     -1.535      0.125      -0.353       0.043
    year_2000                 -0.5602      0.091     -6.164      0.000      -0.738      -0.382
    year_2002                 -0.4908      0.072     -6.811      0.000      -0.632      -0.350
    year_2003                 -0.1934      0.070     -2.748      0.006      -0.331      -0.055
    year_2004                  0.0361      0.066      0.547      0.584      -0.093       0.165
    year_2005                 -0.1362      0.065     -2.098      0.036      -0.263      -0.009
    year_2006                 -0.0076      0.065     -0.117      0.907      -0.134       0.119
    year_2007                  0.1314      0.061      2.137      0.033       0.011       0.252
    year_2008                  0.2690      0.061      4.410      0.000       0.149       0.389
    year_2009                  0.4491      0.060      7.469      0.000       0.331       0.567
    year_2010                  0.5900      0.060      9.819      0.000       0.472       0.708
    year_2011                  0.7727      0.060     12.973      0.000       0.656       0.889
    year_2012                  0.9125      0.059     15.420      0.000       0.796       1.028
    year_2013                  1.0334      0.059     17.626      0.000       0.918       1.148
    year_2014                  1.1540      0.059     19.701      0.000       1.039       1.269
    year_2015                  1.2577      0.059     21.490      0.000       1.143       1.372
    year_2016                  1.3743      0.059     23.485      0.000       1.260       1.489
    year_2017                  1.4472      0.058     24.740      0.000       1.333       1.562
    year_2018                  1.5492      0.059     26.475      0.000       1.435       1.664
    year_2019                  1.7178      0.059     29.344      0.000       1.603       1.833
    year_2020                  1.8417      0.059     31.382      0.000       1.727       1.957
    transmission_Manual       -0.1395      0.003    -54.370      0.000      -0.145      -0.135
    transmission_Other        -0.0597      0.083     -0.724      0.469      -0.222       0.102
    transmission_Semi-Auto     0.0006      0.002      0.237      0.813      -0.004       0.005
    fuelType_Electric          0.7816      0.108      7.250      0.000       0.570       0.993
    fuelType_Hybrid            0.2548      0.005     47.419      0.000       0.244       0.265
    fuelType_Other             0.1584      0.017      9.068      0.000       0.124       0.193
    fuelType_Petrol           -0.0239      0.002    -11.310      0.000      -0.028      -0.020
    brand_bmw                 -0.1106      0.003    -35.000      0.000      -0.117      -0.104
    brand_cclass               0.1309      0.005     25.071      0.000       0.121       0.141
    brand_ford                -0.2543      0.003    -84.688      0.000      -0.260      -0.248
    brand_hyundi              -0.3611      0.004    -90.265      0.000      -0.369      -0.353
    brand_merc                -0.0283      0.003     -9.264      0.000      -0.034      -0.022
    brand_skoda               -0.2866      0.004    -78.991      0.000      -0.294      -0.280
    brand_toyota              -0.4130      0.004   -104.686      0.000      -0.421      -0.405
    price_contribution        -0.1876      0.006    -30.361      0.000      -0.200      -0.176
    ==============================================================================
    Omnibus:                     4315.110   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20475.406
    Skew:                           0.268   Prob(JB):                         0.00
    Kurtosis:                       6.017   Cond. No.                     1.14e+07
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.14e+07. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing year_2006 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.882
    Model:                            OLS   Adj. R-squared:                  0.882
    Method:                 Least Squares   F-statistic:                     9506.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:43:31   Log-Likelihood:                 14242.
    No. Observations:               52335   AIC:                        -2.840e+04
    Df Residuals:                   52293   BIC:                        -2.803e+04
    Df Model:                          41                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                      7.9299      0.027    293.860      0.000       7.877       7.983
    mileage                -5.057e-06   6.09e-08    -82.988      0.000   -5.18e-06   -4.94e-06
    tax                        0.0005   1.68e-05     28.772      0.000       0.000       0.001
    mpg                       -0.0010   6.42e-05    -16.166      0.000      -0.001      -0.001
    engineSize                 0.3544      0.002    172.511      0.000       0.350       0.358
    year_1997                  0.4972      0.186      2.672      0.008       0.132       0.862
    year_1998                  0.0618      0.086      0.717      0.473      -0.107       0.231
    year_1999                 -0.1489      0.086     -1.727      0.084      -0.318       0.020
    year_2000                 -0.5541      0.074     -7.478      0.000      -0.699      -0.409
    year_2002                 -0.4846      0.049     -9.850      0.000      -0.581      -0.388
    year_2003                 -0.1872      0.047     -4.013      0.000      -0.279      -0.096
    year_2004                  0.0422      0.040      1.064      0.287      -0.036       0.120
    year_2005                 -0.1301      0.038     -3.427      0.001      -0.204      -0.056
    year_2007                  0.1376      0.032      4.333      0.000       0.075       0.200
    year_2008                  0.2752      0.031      8.945      0.000       0.215       0.335
    year_2009                  0.4553      0.029     15.694      0.000       0.398       0.512
    year_2010                  0.5962      0.029     20.611      0.000       0.539       0.653
    year_2011                  0.7788      0.028     27.993      0.000       0.724       0.833
    year_2012                  0.9186      0.027     34.023      0.000       0.866       0.972
    year_2013                  1.0395      0.026     40.309      0.000       0.989       1.090
    year_2014                  1.1602      0.026     45.165      0.000       1.110       1.211
    year_2015                  1.2639      0.026     49.417      0.000       1.214       1.314
    year_2016                  1.3804      0.026     53.997      0.000       1.330       1.431
    year_2017                  1.4534      0.026     56.903      0.000       1.403       1.503
    year_2018                  1.5554      0.026     60.749      0.000       1.505       1.606
    year_2019                  1.7239      0.026     67.136      0.000       1.674       1.774
    year_2020                  1.8478      0.026     71.016      0.000       1.797       1.899
    transmission_Manual       -0.1395      0.003    -54.375      0.000      -0.145      -0.135
    transmission_Other        -0.0597      0.083     -0.724      0.469      -0.222       0.102
    transmission_Semi-Auto     0.0006      0.002      0.235      0.814      -0.004       0.005
    fuelType_Electric          0.7816      0.108      7.250      0.000       0.570       0.993
    fuelType_Hybrid            0.2548      0.005     47.420      0.000       0.244       0.265
    fuelType_Other             0.1584      0.017      9.068      0.000       0.124       0.193
    fuelType_Petrol           -0.0239      0.002    -11.309      0.000      -0.028      -0.020
    brand_bmw                 -0.1106      0.003    -35.000      0.000      -0.117      -0.104
    brand_cclass               0.1309      0.005     25.071      0.000       0.121       0.141
    brand_ford                -0.2543      0.003    -84.689      0.000      -0.260      -0.248
    brand_hyundi              -0.3611      0.004    -90.266      0.000      -0.369      -0.353
    brand_merc                -0.0283      0.003     -9.263      0.000      -0.034      -0.022
    brand_skoda               -0.2866      0.004    -78.992      0.000      -0.294      -0.280
    brand_toyota              -0.4130      0.004   -104.687      0.000      -0.421      -0.405
    price_contribution        -0.1876      0.006    -30.361      0.000      -0.200      -0.176
    ==============================================================================
    Omnibus:                     4319.023   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20508.975
    Skew:                           0.269   Prob(JB):                         0.00
    Kurtosis:                       6.019   Cond. No.                     7.32e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.32e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing transmission_Semi-Auto 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.882
    Model:                            OLS   Adj. R-squared:                  0.882
    Method:                 Least Squares   F-statistic:                     9744.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:43:31   Log-Likelihood:                 14242.
    No. Observations:               52335   AIC:                        -2.840e+04
    Df Residuals:                   52294   BIC:                        -2.804e+04
    Df Model:                          40                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    const                   7.9301      0.027    294.060      0.000       7.877       7.983
    mileage             -5.057e-06   6.09e-08    -83.071      0.000   -5.18e-06   -4.94e-06
    tax                     0.0005   1.68e-05     28.775      0.000       0.000       0.001
    mpg                    -0.0010   6.42e-05    -16.165      0.000      -0.001      -0.001
    engineSize              0.3543      0.002    172.513      0.000       0.350       0.358
    year_1997               0.4970      0.186      2.671      0.008       0.132       0.862
    year_1998               0.0618      0.086      0.716      0.474      -0.107       0.231
    year_1999              -0.1489      0.086     -1.727      0.084      -0.318       0.020
    year_2000              -0.5541      0.074     -7.479      0.000      -0.699      -0.409
    year_2002              -0.4847      0.049     -9.852      0.000      -0.581      -0.388
    year_2003              -0.1872      0.047     -4.013      0.000      -0.279      -0.096
    year_2004               0.0422      0.040      1.064      0.287      -0.036       0.120
    year_2005              -0.1300      0.038     -3.426      0.001      -0.204      -0.056
    year_2007               0.1376      0.032      4.334      0.000       0.075       0.200
    year_2008               0.2753      0.031      8.948      0.000       0.215       0.336
    year_2009               0.4553      0.029     15.697      0.000       0.398       0.512
    year_2010               0.5962      0.029     20.613      0.000       0.540       0.653
    year_2011               0.7788      0.028     27.994      0.000       0.724       0.833
    year_2012               0.9186      0.027     34.023      0.000       0.866       0.972
    year_2013               1.0396      0.026     40.315      0.000       0.989       1.090
    year_2014               1.1602      0.026     45.173      0.000       1.110       1.211
    year_2015               1.2640      0.026     49.425      0.000       1.214       1.314
    year_2016               1.3805      0.026     54.006      0.000       1.330       1.431
    year_2017               1.4535      0.026     56.916      0.000       1.403       1.504
    year_2018               1.5555      0.026     60.760      0.000       1.505       1.606
    year_2019               1.7240      0.026     67.149      0.000       1.674       1.774
    year_2020               1.8480      0.026     71.049      0.000       1.797       1.899
    transmission_Manual    -0.1398      0.002    -61.806      0.000      -0.144      -0.135
    transmission_Other     -0.0599      0.083     -0.726      0.468      -0.222       0.102
    fuelType_Electric       0.7813      0.108      7.247      0.000       0.570       0.993
    fuelType_Hybrid         0.2545      0.005     48.039      0.000       0.244       0.265
    fuelType_Other          0.1581      0.017      9.072      0.000       0.124       0.192
    fuelType_Petrol        -0.0239      0.002    -11.308      0.000      -0.028      -0.020
    brand_bmw              -0.1106      0.003    -35.006      0.000      -0.117      -0.104
    brand_cclass            0.1310      0.005     25.115      0.000       0.121       0.141
    brand_ford             -0.2544      0.003    -84.762      0.000      -0.260      -0.248
    brand_hyundi           -0.3611      0.004    -90.277      0.000      -0.369      -0.353
    brand_merc             -0.0283      0.003     -9.261      0.000      -0.034      -0.022
    brand_skoda            -0.2866      0.004    -79.005      0.000      -0.294      -0.280
    brand_toyota           -0.4131      0.004   -104.917      0.000      -0.421      -0.405
    price_contribution     -0.1877      0.006    -30.384      0.000      -0.200      -0.176
    ==============================================================================
    Omnibus:                     4317.353   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20498.681
    Skew:                           0.268   Prob(JB):                         0.00
    Kurtosis:                       6.019   Cond. No.                     7.32e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.32e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing year_1998 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.882
    Model:                            OLS   Adj. R-squared:                  0.882
    Method:                 Least Squares   F-statistic:                     9994.
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:43:31   Log-Likelihood:                 14242.
    No. Observations:               52335   AIC:                        -2.840e+04
    Df Residuals:                   52295   BIC:                        -2.805e+04
    Df Model:                          39                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    const                   7.9353      0.026    305.404      0.000       7.884       7.986
    mileage             -5.057e-06   6.09e-08    -83.069      0.000   -5.18e-06   -4.94e-06
    tax                     0.0005   1.68e-05     28.767      0.000       0.000       0.001
    mpg                    -0.0010   6.42e-05    -16.163      0.000      -0.001      -0.001
    engineSize              0.3544      0.002    172.577      0.000       0.350       0.358
    year_1997               0.4918      0.186      2.644      0.008       0.127       0.856
    year_1999              -0.1541      0.086     -1.794      0.073      -0.322       0.014
    year_2000              -0.5594      0.074     -7.587      0.000      -0.704      -0.415
    year_2002              -0.4899      0.049    -10.071      0.000      -0.585      -0.395
    year_2003              -0.1925      0.046     -4.177      0.000      -0.283      -0.102
    year_2004               0.0370      0.039      0.948      0.343      -0.039       0.113
    year_2005              -0.1352      0.037     -3.632      0.000      -0.208      -0.062
    year_2007               0.1324      0.031      4.285      0.000       0.072       0.193
    year_2008               0.2700      0.030      9.037      0.000       0.211       0.329
    year_2009               0.4501      0.028     16.035      0.000       0.395       0.505
    year_2010               0.5910      0.028     21.119      0.000       0.536       0.646
    year_2011               0.7736      0.027     28.821      0.000       0.721       0.826
    year_2012               0.9134      0.026     35.145      0.000       0.862       0.964
    year_2013               1.0343      0.025     41.833      0.000       0.986       1.083
    year_2014               1.1550      0.025     46.916      0.000       1.107       1.203
    year_2015               1.2587      0.025     51.371      0.000       1.211       1.307
    year_2016               1.3753      0.024     56.154      0.000       1.327       1.423
    year_2017               1.4482      0.024     59.189      0.000       1.400       1.496
    year_2018               1.5503      0.025     63.179      0.000       1.502       1.598
    year_2019               1.7188      0.025     69.827      0.000       1.671       1.767
    year_2020               1.8428      0.025     73.813      0.000       1.794       1.892
    transmission_Manual    -0.1398      0.002    -61.806      0.000      -0.144      -0.135
    transmission_Other     -0.0599      0.083     -0.726      0.468      -0.222       0.102
    fuelType_Electric       0.7813      0.108      7.247      0.000       0.570       0.993
    fuelType_Hybrid         0.2545      0.005     48.037      0.000       0.244       0.265
    fuelType_Other          0.1581      0.017      9.072      0.000       0.124       0.192
    fuelType_Petrol        -0.0239      0.002    -11.299      0.000      -0.028      -0.020
    brand_bmw              -0.1106      0.003    -35.009      0.000      -0.117      -0.104
    brand_cclass            0.1310      0.005     25.116      0.000       0.121       0.141
    brand_ford             -0.2544      0.003    -84.760      0.000      -0.260      -0.248
    brand_hyundi           -0.3611      0.004    -90.275      0.000      -0.369      -0.353
    brand_merc             -0.0283      0.003     -9.257      0.000      -0.034      -0.022
    brand_skoda            -0.2866      0.004    -79.003      0.000      -0.294      -0.280
    brand_toyota           -0.4130      0.004   -104.915      0.000      -0.421      -0.405
    price_contribution     -0.1877      0.006    -30.385      0.000      -0.200      -0.176
    ==============================================================================
    Omnibus:                     4329.620   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20588.825
    Skew:                           0.270   Prob(JB):                         0.00
    Kurtosis:                       6.025   Cond. No.                     7.30e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.3e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing transmission_Other 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.882
    Model:                            OLS   Adj. R-squared:                  0.882
    Method:                 Least Squares   F-statistic:                 1.026e+04
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:43:31   Log-Likelihood:                 14242.
    No. Observations:               52335   AIC:                        -2.841e+04
    Df Residuals:                   52296   BIC:                        -2.806e+04
    Df Model:                          38                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    const                   7.9352      0.026    305.404      0.000       7.884       7.986
    mileage             -5.057e-06   6.09e-08    -83.067      0.000   -5.18e-06   -4.94e-06
    tax                     0.0005   1.68e-05     28.770      0.000       0.000       0.001
    mpg                    -0.0010   6.42e-05    -16.161      0.000      -0.001      -0.001
    engineSize              0.3544      0.002    172.577      0.000       0.350       0.358
    year_1997               0.4918      0.186      2.644      0.008       0.127       0.856
    year_1999              -0.1541      0.086     -1.794      0.073      -0.322       0.014
    year_2000              -0.5594      0.074     -7.587      0.000      -0.704      -0.415
    year_2002              -0.4899      0.049    -10.070      0.000      -0.585      -0.395
    year_2003              -0.1924      0.046     -4.177      0.000      -0.283      -0.102
    year_2004               0.0370      0.039      0.948      0.343      -0.039       0.113
    year_2005              -0.1352      0.037     -3.632      0.000      -0.208      -0.062
    year_2007               0.1324      0.031      4.285      0.000       0.072       0.193
    year_2008               0.2700      0.030      9.037      0.000       0.211       0.329
    year_2009               0.4501      0.028     16.035      0.000       0.395       0.505
    year_2010               0.5910      0.028     21.119      0.000       0.536       0.646
    year_2011               0.7736      0.027     28.821      0.000       0.721       0.826
    year_2012               0.9133      0.026     35.140      0.000       0.862       0.964
    year_2013               1.0344      0.025     41.834      0.000       0.986       1.083
    year_2014               1.1550      0.025     46.917      0.000       1.107       1.203
    year_2015               1.2587      0.025     51.371      0.000       1.211       1.307
    year_2016               1.3753      0.024     56.155      0.000       1.327       1.423
    year_2017               1.4483      0.024     59.189      0.000       1.400       1.496
    year_2018               1.5503      0.025     63.180      0.000       1.502       1.598
    year_2019               1.7188      0.025     69.829      0.000       1.671       1.767
    year_2020               1.8428      0.025     73.815      0.000       1.794       1.892
    transmission_Manual    -0.1398      0.002    -61.803      0.000      -0.144      -0.135
    fuelType_Electric       0.7813      0.108      7.247      0.000       0.570       0.993
    fuelType_Hybrid         0.2545      0.005     48.032      0.000       0.244       0.265
    fuelType_Other          0.1581      0.017      9.072      0.000       0.124       0.192
    fuelType_Petrol        -0.0239      0.002    -11.299      0.000      -0.028      -0.020
    brand_bmw              -0.1106      0.003    -35.008      0.000      -0.117      -0.104
    brand_cclass            0.1310      0.005     25.114      0.000       0.121       0.141
    brand_ford             -0.2544      0.003    -84.762      0.000      -0.260      -0.248
    brand_hyundi           -0.3611      0.004    -90.293      0.000      -0.369      -0.353
    brand_merc             -0.0283      0.003     -9.259      0.000      -0.034      -0.022
    brand_skoda            -0.2866      0.004    -79.004      0.000      -0.294      -0.280
    brand_toyota           -0.4130      0.004   -104.915      0.000      -0.421      -0.405
    price_contribution     -0.1877      0.006    -30.383      0.000      -0.200      -0.176
    ==============================================================================
    Omnibus:                     4329.316   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20587.010
    Skew:                           0.269   Prob(JB):                         0.00
    Kurtosis:                       6.025   Cond. No.                     7.30e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.3e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    After removing year_2004 
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.882
    Model:                            OLS   Adj. R-squared:                  0.882
    Method:                 Least Squares   F-statistic:                 1.053e+04
    Date:                Fri, 16 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:43:31   Log-Likelihood:                 14241.
    No. Observations:               52335   AIC:                        -2.841e+04
    Df Residuals:                   52297   BIC:                        -2.807e+04
    Df Model:                          37                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    const                   7.9492      0.021    371.280      0.000       7.907       7.991
    mileage             -5.056e-06   6.09e-08    -83.062      0.000   -5.18e-06   -4.94e-06
    tax                     0.0005   1.68e-05     28.771      0.000       0.000       0.001
    mpg                    -0.0010   6.42e-05    -16.170      0.000      -0.001      -0.001
    engineSize              0.3544      0.002    172.575      0.000       0.350       0.358
    year_1997               0.4778      0.185      2.577      0.010       0.114       0.841
    year_1999              -0.1681      0.085     -1.987      0.047      -0.334      -0.002
    year_2000              -0.5734      0.072     -7.938      0.000      -0.715      -0.432
    year_2002              -0.5039      0.046    -10.871      0.000      -0.595      -0.413
    year_2003              -0.2065      0.044     -4.731      0.000      -0.292      -0.121
    year_2005              -0.1492      0.034     -4.366      0.000      -0.216      -0.082
    year_2007               0.1184      0.027      4.362      0.000       0.065       0.172
    year_2008               0.2560      0.026      9.858      0.000       0.205       0.307
    year_2009               0.4361      0.024     18.269      0.000       0.389       0.483
    year_2010               0.5770      0.024     24.274      0.000       0.530       0.624
    year_2011               0.7596      0.022     33.878      0.000       0.716       0.804
    year_2012               0.8993      0.021     42.022      0.000       0.857       0.941
    year_2013               1.0204      0.020     51.391      0.000       0.981       1.059
    year_2014               1.1411      0.020     57.843      0.000       1.102       1.180
    year_2015               1.2448      0.020     63.550      0.000       1.206       1.283
    year_2016               1.3613      0.020     69.532      0.000       1.323       1.400
    year_2017               1.4343      0.020     73.344      0.000       1.396       1.473
    year_2018               1.5363      0.020     78.199      0.000       1.498       1.575
    year_2019               1.7049      0.020     86.335      0.000       1.666       1.744
    year_2020               1.8289      0.020     90.611      0.000       1.789       1.868
    transmission_Manual    -0.1398      0.002    -61.801      0.000      -0.144      -0.135
    fuelType_Electric       0.7814      0.108      7.249      0.000       0.570       0.993
    fuelType_Hybrid         0.2545      0.005     48.036      0.000       0.244       0.265
    fuelType_Other          0.1581      0.017      9.073      0.000       0.124       0.192
    fuelType_Petrol        -0.0239      0.002    -11.295      0.000      -0.028      -0.020
    brand_bmw              -0.1105      0.003    -35.002      0.000      -0.117      -0.104
    brand_cclass            0.1310      0.005     25.117      0.000       0.121       0.141
    brand_ford             -0.2544      0.003    -84.764      0.000      -0.260      -0.248
    brand_hyundi           -0.3611      0.004    -90.293      0.000      -0.369      -0.353
    brand_merc             -0.0282      0.003     -9.251      0.000      -0.034      -0.022
    brand_skoda            -0.2866      0.004    -79.002      0.000      -0.294      -0.280
    brand_toyota           -0.4130      0.004   -104.912      0.000      -0.421      -0.405
    price_contribution     -0.1877      0.006    -30.390      0.000      -0.200      -0.176
    ==============================================================================
    Omnibus:                     4336.554   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20622.318
    Skew:                           0.271   Prob(JB):                         0.00
    Kurtosis:                       6.027   Cond. No.                     7.26e+06
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.26e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#final model
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.882</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.882</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.053e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 16 Feb 2024</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>13:43:44</td>     <th>  Log-Likelihood:    </th>  <td>  14241.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 52335</td>      <th>  AIC:               </th> <td>-2.841e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 52297</td>      <th>  BIC:               </th> <td>-2.807e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    37</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>               <td>    7.9492</td> <td>    0.021</td> <td>  371.280</td> <td> 0.000</td> <td>    7.907</td> <td>    7.991</td>
</tr>
<tr>
  <th>mileage</th>             <td>-5.056e-06</td> <td> 6.09e-08</td> <td>  -83.062</td> <td> 0.000</td> <td>-5.18e-06</td> <td>-4.94e-06</td>
</tr>
<tr>
  <th>tax</th>                 <td>    0.0005</td> <td> 1.68e-05</td> <td>   28.771</td> <td> 0.000</td> <td>    0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>mpg</th>                 <td>   -0.0010</td> <td> 6.42e-05</td> <td>  -16.170</td> <td> 0.000</td> <td>   -0.001</td> <td>   -0.001</td>
</tr>
<tr>
  <th>engineSize</th>          <td>    0.3544</td> <td>    0.002</td> <td>  172.575</td> <td> 0.000</td> <td>    0.350</td> <td>    0.358</td>
</tr>
<tr>
  <th>year_1997</th>           <td>    0.4778</td> <td>    0.185</td> <td>    2.577</td> <td> 0.010</td> <td>    0.114</td> <td>    0.841</td>
</tr>
<tr>
  <th>year_1999</th>           <td>   -0.1681</td> <td>    0.085</td> <td>   -1.987</td> <td> 0.047</td> <td>   -0.334</td> <td>   -0.002</td>
</tr>
<tr>
  <th>year_2000</th>           <td>   -0.5734</td> <td>    0.072</td> <td>   -7.938</td> <td> 0.000</td> <td>   -0.715</td> <td>   -0.432</td>
</tr>
<tr>
  <th>year_2002</th>           <td>   -0.5039</td> <td>    0.046</td> <td>  -10.871</td> <td> 0.000</td> <td>   -0.595</td> <td>   -0.413</td>
</tr>
<tr>
  <th>year_2003</th>           <td>   -0.2065</td> <td>    0.044</td> <td>   -4.731</td> <td> 0.000</td> <td>   -0.292</td> <td>   -0.121</td>
</tr>
<tr>
  <th>year_2005</th>           <td>   -0.1492</td> <td>    0.034</td> <td>   -4.366</td> <td> 0.000</td> <td>   -0.216</td> <td>   -0.082</td>
</tr>
<tr>
  <th>year_2007</th>           <td>    0.1184</td> <td>    0.027</td> <td>    4.362</td> <td> 0.000</td> <td>    0.065</td> <td>    0.172</td>
</tr>
<tr>
  <th>year_2008</th>           <td>    0.2560</td> <td>    0.026</td> <td>    9.858</td> <td> 0.000</td> <td>    0.205</td> <td>    0.307</td>
</tr>
<tr>
  <th>year_2009</th>           <td>    0.4361</td> <td>    0.024</td> <td>   18.269</td> <td> 0.000</td> <td>    0.389</td> <td>    0.483</td>
</tr>
<tr>
  <th>year_2010</th>           <td>    0.5770</td> <td>    0.024</td> <td>   24.274</td> <td> 0.000</td> <td>    0.530</td> <td>    0.624</td>
</tr>
<tr>
  <th>year_2011</th>           <td>    0.7596</td> <td>    0.022</td> <td>   33.878</td> <td> 0.000</td> <td>    0.716</td> <td>    0.804</td>
</tr>
<tr>
  <th>year_2012</th>           <td>    0.8993</td> <td>    0.021</td> <td>   42.022</td> <td> 0.000</td> <td>    0.857</td> <td>    0.941</td>
</tr>
<tr>
  <th>year_2013</th>           <td>    1.0204</td> <td>    0.020</td> <td>   51.391</td> <td> 0.000</td> <td>    0.981</td> <td>    1.059</td>
</tr>
<tr>
  <th>year_2014</th>           <td>    1.1411</td> <td>    0.020</td> <td>   57.843</td> <td> 0.000</td> <td>    1.102</td> <td>    1.180</td>
</tr>
<tr>
  <th>year_2015</th>           <td>    1.2448</td> <td>    0.020</td> <td>   63.550</td> <td> 0.000</td> <td>    1.206</td> <td>    1.283</td>
</tr>
<tr>
  <th>year_2016</th>           <td>    1.3613</td> <td>    0.020</td> <td>   69.532</td> <td> 0.000</td> <td>    1.323</td> <td>    1.400</td>
</tr>
<tr>
  <th>year_2017</th>           <td>    1.4343</td> <td>    0.020</td> <td>   73.344</td> <td> 0.000</td> <td>    1.396</td> <td>    1.473</td>
</tr>
<tr>
  <th>year_2018</th>           <td>    1.5363</td> <td>    0.020</td> <td>   78.199</td> <td> 0.000</td> <td>    1.498</td> <td>    1.575</td>
</tr>
<tr>
  <th>year_2019</th>           <td>    1.7049</td> <td>    0.020</td> <td>   86.335</td> <td> 0.000</td> <td>    1.666</td> <td>    1.744</td>
</tr>
<tr>
  <th>year_2020</th>           <td>    1.8289</td> <td>    0.020</td> <td>   90.611</td> <td> 0.000</td> <td>    1.789</td> <td>    1.868</td>
</tr>
<tr>
  <th>transmission_Manual</th> <td>   -0.1398</td> <td>    0.002</td> <td>  -61.801</td> <td> 0.000</td> <td>   -0.144</td> <td>   -0.135</td>
</tr>
<tr>
  <th>fuelType_Electric</th>   <td>    0.7814</td> <td>    0.108</td> <td>    7.249</td> <td> 0.000</td> <td>    0.570</td> <td>    0.993</td>
</tr>
<tr>
  <th>fuelType_Hybrid</th>     <td>    0.2545</td> <td>    0.005</td> <td>   48.036</td> <td> 0.000</td> <td>    0.244</td> <td>    0.265</td>
</tr>
<tr>
  <th>fuelType_Other</th>      <td>    0.1581</td> <td>    0.017</td> <td>    9.073</td> <td> 0.000</td> <td>    0.124</td> <td>    0.192</td>
</tr>
<tr>
  <th>fuelType_Petrol</th>     <td>   -0.0239</td> <td>    0.002</td> <td>  -11.295</td> <td> 0.000</td> <td>   -0.028</td> <td>   -0.020</td>
</tr>
<tr>
  <th>brand_bmw</th>           <td>   -0.1105</td> <td>    0.003</td> <td>  -35.002</td> <td> 0.000</td> <td>   -0.117</td> <td>   -0.104</td>
</tr>
<tr>
  <th>brand_cclass</th>        <td>    0.1310</td> <td>    0.005</td> <td>   25.117</td> <td> 0.000</td> <td>    0.121</td> <td>    0.141</td>
</tr>
<tr>
  <th>brand_ford</th>          <td>   -0.2544</td> <td>    0.003</td> <td>  -84.764</td> <td> 0.000</td> <td>   -0.260</td> <td>   -0.248</td>
</tr>
<tr>
  <th>brand_hyundi</th>        <td>   -0.3611</td> <td>    0.004</td> <td>  -90.293</td> <td> 0.000</td> <td>   -0.369</td> <td>   -0.353</td>
</tr>
<tr>
  <th>brand_merc</th>          <td>   -0.0282</td> <td>    0.003</td> <td>   -9.251</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.022</td>
</tr>
<tr>
  <th>brand_skoda</th>         <td>   -0.2866</td> <td>    0.004</td> <td>  -79.002</td> <td> 0.000</td> <td>   -0.294</td> <td>   -0.280</td>
</tr>
<tr>
  <th>brand_toyota</th>        <td>   -0.4130</td> <td>    0.004</td> <td> -104.912</td> <td> 0.000</td> <td>   -0.421</td> <td>   -0.405</td>
</tr>
<tr>
  <th>price_contribution</th>  <td>   -0.1877</td> <td>    0.006</td> <td>  -30.390</td> <td> 0.000</td> <td>   -0.200</td> <td>   -0.176</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>4336.554</td> <th>  Durbin-Watson:     </th> <td>   2.000</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>20622.318</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.271</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.027</td>  <th>  Cond. No.          </th> <td>7.26e+06</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 7.26e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
#test r-square 0.83 higher than 0.81 (r2 value without target transformation)
X_test_ols = sm.add_constant(X_test[cols])
r2_score(y_test.reset_index(drop = True)['price'],np.exp(model.predict(X_test_ols)).fillna(0))
```




    0.8327725785236579




```python
#model coefficients (we cannot interpret on basis of feature importance just we obs coefficient values as these are not scalled to same scale)
model.params
```




    const                  7.949162
    mileage               -0.000005
    tax                    0.000483
    mpg                   -0.001038
    engineSize             0.354371
    year_1997              0.477764
    year_1999             -0.168123
    year_2000             -0.573384
    year_2002             -0.503933
    year_2003             -0.206459
    year_2005             -0.149249
    year_2007              0.118387
    year_2008              0.256012
    year_2009              0.436082
    year_2010              0.576964
    year_2011              0.759618
    year_2012              0.899272
    year_2013              1.020392
    year_2014              1.141056
    year_2015              1.244756
    year_2016              1.361346
    year_2017              1.434321
    year_2018              1.536349
    year_2019              1.704890
    year_2020              1.828877
    transmission_Manual   -0.139805
    fuelType_Electric      0.781426
    fuelType_Hybrid        0.254502
    fuelType_Other         0.158121
    fuelType_Petrol       -0.023863
    brand_bmw             -0.110542
    brand_cclass           0.131005
    brand_ford            -0.254364
    brand_hyundi          -0.361132
    brand_merc            -0.028232
    brand_skoda           -0.286616
    brand_toyota          -0.413018
    price_contribution    -0.187696
    dtype: float64




```python
#checking OLS method assumptions

# 1. No autocorrelation of residuals -> Durbin-Watson value <2 suggests there is no autocorrelation of residuals

# 2. Homoscedasticity -> by plotting residuals vs fitted values
plt.scatter(np.exp(model.fittedvalues), np.exp(model.fittedvalues) - y_train.reset_index()['price'])
plt.show()

#thers is still hetroscedasticity present maybe we would need more advance model , can also try polynomial transformation of independent variables
```


    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_63_0.png)
    



```python
#3. Regarding normal distribution of errors its slightly better than before with skew value ~0.163 (refer to summary()) 
(np.exp(model.fittedvalues) - y_train.reset_index()['price']).hist()
```




    <Axes: >




    
![png](https://dakshjain97.github.io/assets/img/car-price-prediction-ols_files/car-price-prediction-ols_64_1.png)
    

