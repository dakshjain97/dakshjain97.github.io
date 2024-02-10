---
layout: post
title: Prediction of returns for fashion orders in ecommerce
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [classification, supervised learning, ecommerce]
author: Daksh Jain
---

```python
#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import pickle
import warnings
```


```python
#Jupyter Notebook Settings
warnings.filterwarnings('ignore')
pd.options.display.max_columns=None
```


```python
#Reading Data for Training
data=pd.read_csv('../Input_Data/TrainingData_V1.csv')
```


```python
#Checking Shape of Data
data.shape
```




    (79945, 14)




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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>22-06-2016</td>
      <td>27-06-2016</td>
      <td>643</td>
      <td>38</td>
      <td>navy</td>
      <td>30</td>
      <td>49.9</td>
      <td>30822</td>
      <td>Mrs</td>
      <td>17-04-1969</td>
      <td>1013</td>
      <td>23-06-2016</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>22-06-2016</td>
      <td>27-06-2016</td>
      <td>195</td>
      <td>xxl</td>
      <td>grey</td>
      <td>46</td>
      <td>19.9</td>
      <td>30823</td>
      <td>Mrs</td>
      <td>22-04-1970</td>
      <td>1001</td>
      <td>15-03-2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>22-06-2016</td>
      <td>05-07-2016</td>
      <td>25</td>
      <td>xxl</td>
      <td>grey</td>
      <td>5</td>
      <td>79.9</td>
      <td>30823</td>
      <td>Mrs</td>
      <td>22-04-1970</td>
      <td>1001</td>
      <td>15-03-2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>23-06-2016</td>
      <td>26-06-2016</td>
      <td>173</td>
      <td>m</td>
      <td>brown</td>
      <td>20</td>
      <td>19.9</td>
      <td>17234</td>
      <td>Mrs</td>
      <td>09-01-1960</td>
      <td>1013</td>
      <td>17-02-2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43</td>
      <td>23-06-2016</td>
      <td>26-06-2016</td>
      <td>394</td>
      <td>40</td>
      <td>black</td>
      <td>44</td>
      <td>90.0</td>
      <td>30827</td>
      <td>Mrs</td>
      <td>NaN</td>
      <td>1006</td>
      <td>09-02-2016</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Checking Data Types
data.dtypes
```




    order_item_id      int64
    order_date        object
    delivery_date     object
    item_id            int64
    item_size         object
    item_color        object
    brand_id           int64
    item_price       float64
    user_id            int64
    user_title        object
    user_dob          object
    user_state         int64
    user_reg_date     object
    return             int64
    dtype: object




```python
#Converting date objects to datetime
data['order_date']=pd.to_datetime(data['order_date'],errors='coerce',format='%d-%m-%Y')
data['delivery_date']=pd.to_datetime(data['delivery_date'],errors='coerce',format='%d-%m-%Y')
data['user_dob']=pd.to_datetime(data['user_dob'],errors='coerce',format='%d-%m-%Y')
data['user_reg_date']=pd.to_datetime(data['user_reg_date'],errors='coerce',format='%d-%m-%Y')
```


```python
data.dtypes
```




    order_item_id             int64
    order_date       datetime64[ns]
    delivery_date    datetime64[ns]
    item_id                   int64
    item_size                object
    item_color               object
    brand_id                  int64
    item_price              float64
    user_id                   int64
    user_title               object
    user_dob         datetime64[ns]
    user_state                int64
    user_reg_date    datetime64[ns]
    return                    int64
    dtype: object



# Checking Data Anomalies/Data Cleaning


```python
#Sorting Data by primary key order item id
data.sort_values(['order_item_id']).reset_index(drop=True,inplace=True)
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2016-06-22</td>
      <td>2016-06-27</td>
      <td>643</td>
      <td>38</td>
      <td>navy</td>
      <td>30</td>
      <td>49.9</td>
      <td>30822</td>
      <td>Mrs</td>
      <td>1969-04-17</td>
      <td>1013</td>
      <td>2016-06-23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>2016-06-22</td>
      <td>2016-06-27</td>
      <td>195</td>
      <td>xxl</td>
      <td>grey</td>
      <td>46</td>
      <td>19.9</td>
      <td>30823</td>
      <td>Mrs</td>
      <td>1970-04-22</td>
      <td>1001</td>
      <td>2015-03-15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>2016-06-22</td>
      <td>2016-07-05</td>
      <td>25</td>
      <td>xxl</td>
      <td>grey</td>
      <td>5</td>
      <td>79.9</td>
      <td>30823</td>
      <td>Mrs</td>
      <td>1970-04-22</td>
      <td>1001</td>
      <td>2015-03-15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>2016-06-23</td>
      <td>2016-06-26</td>
      <td>173</td>
      <td>m</td>
      <td>brown</td>
      <td>20</td>
      <td>19.9</td>
      <td>17234</td>
      <td>Mrs</td>
      <td>1960-01-09</td>
      <td>1013</td>
      <td>2015-02-17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43</td>
      <td>2016-06-23</td>
      <td>2016-06-26</td>
      <td>394</td>
      <td>40</td>
      <td>black</td>
      <td>44</td>
      <td>90.0</td>
      <td>30827</td>
      <td>Mrs</td>
      <td>NaT</td>
      <td>1006</td>
      <td>2016-02-09</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#No Duplicates Found
data.drop_duplicates().shape
```




    (79945, 14)




```python
#No duplicates on primary key
data[data['order_item_id'].duplicated()]
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#Checking cases where user dob could be wrong
data[(data['user_dob']>data['delivery_date']) | (data['user_dob']>data['order_date']) | (data['user_dob']>data['user_reg_date'])]
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#Checking Cases where user registration date could be wrong , for these case ill create a flag
#because user regristraion date cannot be greater than order or delivery date
data[(data['user_reg_date']>data['delivery_date']) | (data['user_reg_date']>data['order_date'])]
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2016-06-22</td>
      <td>2016-06-27</td>
      <td>643</td>
      <td>38</td>
      <td>navy</td>
      <td>30</td>
      <td>49.90</td>
      <td>30822</td>
      <td>Mrs</td>
      <td>1969-04-17</td>
      <td>1013</td>
      <td>2016-06-23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>111</td>
      <td>2016-06-23</td>
      <td>2016-06-27</td>
      <td>166</td>
      <td>38</td>
      <td>white</td>
      <td>6</td>
      <td>69.90</td>
      <td>30837</td>
      <td>Mrs</td>
      <td>1963-02-28</td>
      <td>1002</td>
      <td>2016-06-24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>235</td>
      <td>2016-06-23</td>
      <td>2016-06-27</td>
      <td>262</td>
      <td>40</td>
      <td>black</td>
      <td>12</td>
      <td>69.90</td>
      <td>30856</td>
      <td>Mrs</td>
      <td>1968-06-30</td>
      <td>1002</td>
      <td>2016-06-24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>307</td>
      <td>2016-06-23</td>
      <td>NaT</td>
      <td>68</td>
      <td>m</td>
      <td>purple</td>
      <td>3</td>
      <td>19.90</td>
      <td>30870</td>
      <td>Mrs</td>
      <td>1958-11-20</td>
      <td>1016</td>
      <td>2016-06-24</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>429</td>
      <td>2016-06-23</td>
      <td>2016-06-27</td>
      <td>405</td>
      <td>41</td>
      <td>green</td>
      <td>18</td>
      <td>79.90</td>
      <td>30892</td>
      <td>Mrs</td>
      <td>1966-05-10</td>
      <td>1002</td>
      <td>2016-06-24</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79919</th>
      <td>99640</td>
      <td>2016-09-10</td>
      <td>2016-09-13</td>
      <td>71</td>
      <td>7</td>
      <td>black</td>
      <td>21</td>
      <td>49.95</td>
      <td>48171</td>
      <td>Mrs</td>
      <td>1971-09-12</td>
      <td>1002</td>
      <td>2016-09-11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79930</th>
      <td>99814</td>
      <td>2016-09-11</td>
      <td>2016-09-13</td>
      <td>166</td>
      <td>38</td>
      <td>ocher</td>
      <td>6</td>
      <td>39.90</td>
      <td>48212</td>
      <td>Mrs</td>
      <td>NaT</td>
      <td>1002</td>
      <td>2016-09-12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79937</th>
      <td>99895</td>
      <td>2016-09-11</td>
      <td>2016-09-13</td>
      <td>98</td>
      <td>l</td>
      <td>green</td>
      <td>28</td>
      <td>49.90</td>
      <td>48226</td>
      <td>Mrs</td>
      <td>1973-11-25</td>
      <td>1003</td>
      <td>2016-09-12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79940</th>
      <td>99942</td>
      <td>2016-09-11</td>
      <td>2016-09-12</td>
      <td>39</td>
      <td>41</td>
      <td>blue</td>
      <td>26</td>
      <td>89.90</td>
      <td>48232</td>
      <td>Mrs</td>
      <td>1941-10-24</td>
      <td>1007</td>
      <td>2016-09-12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79941</th>
      <td>99954</td>
      <td>2016-09-11</td>
      <td>NaT</td>
      <td>1498</td>
      <td>42</td>
      <td>green</td>
      <td>6</td>
      <td>59.90</td>
      <td>48234</td>
      <td>Mrs</td>
      <td>1962-10-02</td>
      <td>1007</td>
      <td>2016-09-12</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16688 rows × 14 columns</p>
</div>




```python
#Checking Cases where delivery date could be wrong , for these case ill create a flag
#because order delivery date cannot be less than order date
data[data['delivery_date']<data['order_date']]
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>312</th>
      <td>3141</td>
      <td>2016-06-25</td>
      <td>1994-12-31</td>
      <td>32</td>
      <td>l</td>
      <td>red</td>
      <td>3</td>
      <td>21.90</td>
      <td>598</td>
      <td>Mrs</td>
      <td>1970-05-08</td>
      <td>1003</td>
      <td>2015-02-17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>431</th>
      <td>4377</td>
      <td>2016-06-26</td>
      <td>1994-12-31</td>
      <td>126</td>
      <td>6+</td>
      <td>red</td>
      <td>21</td>
      <td>39.95</td>
      <td>31734</td>
      <td>Mrs</td>
      <td>1965-06-15</td>
      <td>1008</td>
      <td>2016-02-16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>498</th>
      <td>5244</td>
      <td>2016-06-27</td>
      <td>1994-12-31</td>
      <td>27</td>
      <td>5</td>
      <td>brown</td>
      <td>19</td>
      <td>39.90</td>
      <td>31858</td>
      <td>Mr</td>
      <td>NaT</td>
      <td>1008</td>
      <td>2016-06-28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>609</th>
      <td>6348</td>
      <td>2016-06-27</td>
      <td>1994-12-31</td>
      <td>388</td>
      <td>xxl</td>
      <td>black</td>
      <td>3</td>
      <td>49.90</td>
      <td>32010</td>
      <td>Mrs</td>
      <td>1973-06-11</td>
      <td>1001</td>
      <td>2015-02-17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>610</th>
      <td>6350</td>
      <td>2016-06-27</td>
      <td>1994-12-31</td>
      <td>195</td>
      <td>xxl</td>
      <td>curry</td>
      <td>46</td>
      <td>9.90</td>
      <td>32010</td>
      <td>Mrs</td>
      <td>1973-06-11</td>
      <td>1001</td>
      <td>2015-02-17</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79626</th>
      <td>96777</td>
      <td>2016-09-07</td>
      <td>1994-12-31</td>
      <td>1743</td>
      <td>xl</td>
      <td>black</td>
      <td>20</td>
      <td>79.90</td>
      <td>47631</td>
      <td>Mrs</td>
      <td>1967-11-30</td>
      <td>1008</td>
      <td>2016-09-08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79727</th>
      <td>97731</td>
      <td>2016-09-09</td>
      <td>1994-12-31</td>
      <td>2058</td>
      <td>5+</td>
      <td>grey</td>
      <td>4</td>
      <td>180.00</td>
      <td>47791</td>
      <td>Family</td>
      <td>1988-01-08</td>
      <td>1009</td>
      <td>2016-09-10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79851</th>
      <td>98939</td>
      <td>2016-09-10</td>
      <td>1994-12-31</td>
      <td>475</td>
      <td>45</td>
      <td>grey</td>
      <td>1</td>
      <td>99.90</td>
      <td>48031</td>
      <td>Mrs</td>
      <td>1982-11-08</td>
      <td>1006</td>
      <td>2016-09-11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79863</th>
      <td>99042</td>
      <td>2016-09-10</td>
      <td>1994-12-31</td>
      <td>1670</td>
      <td>39</td>
      <td>grey</td>
      <td>1</td>
      <td>184.91</td>
      <td>2649</td>
      <td>Mrs</td>
      <td>1979-02-07</td>
      <td>1001</td>
      <td>2015-06-14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79887</th>
      <td>99368</td>
      <td>2016-09-10</td>
      <td>1994-12-31</td>
      <td>2154</td>
      <td>42</td>
      <td>grey</td>
      <td>7</td>
      <td>19.90</td>
      <td>48120</td>
      <td>Mrs</td>
      <td>1986-02-22</td>
      <td>1008</td>
      <td>2016-09-11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>846 rows × 14 columns</p>
</div>




```python
#No negative price
data[data['item_price']<0]
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#Checking Missing Values
data.isna().sum()
```




    order_item_id       0
    order_date          0
    delivery_date    7436
    item_id             0
    item_size           0
    item_color          0
    brand_id            0
    item_price          0
    user_id             0
    user_title          0
    user_dob         6989
    user_state          0
    user_reg_date       0
    return              0
    dtype: int64



# Feature Engineering


```python
#Creating Flag For Data Anomalies
#Invalid User Reg Date is a flag to check if registration date of user is not valid
#Invalid Delivery Date is a flag to check if delivery data of product is invalid
data['Invalid_User_Reg_Date_Flag']=0
data['Invalid_Delivery_Date']=0
for i in range(0,len(data)):
    if ((data.at[i,'user_reg_date']>data.at[i,'delivery_date']) or (data.at[i,'user_reg_date']>data.at[i,'order_date'])):
        data.at[i,'Invalid_User_Reg_Date_Flag']=1
    if data.at[i,'delivery_date']<data.at[i,'order_date']:
        data.at[i,'Invalid_Delivery_Date']=1
```


```python
#Data with Flag for invalid user reg date
data[(data['user_reg_date']>data['delivery_date']) | (data['user_reg_date']>data['order_date'])]
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
      <th>Invalid_User_Reg_Date_Flag</th>
      <th>Invalid_Delivery_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2016-06-22</td>
      <td>2016-06-27</td>
      <td>643</td>
      <td>38</td>
      <td>navy</td>
      <td>30</td>
      <td>49.90</td>
      <td>30822</td>
      <td>Mrs</td>
      <td>1969-04-17</td>
      <td>1013</td>
      <td>2016-06-23</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>111</td>
      <td>2016-06-23</td>
      <td>2016-06-27</td>
      <td>166</td>
      <td>38</td>
      <td>white</td>
      <td>6</td>
      <td>69.90</td>
      <td>30837</td>
      <td>Mrs</td>
      <td>1963-02-28</td>
      <td>1002</td>
      <td>2016-06-24</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>235</td>
      <td>2016-06-23</td>
      <td>2016-06-27</td>
      <td>262</td>
      <td>40</td>
      <td>black</td>
      <td>12</td>
      <td>69.90</td>
      <td>30856</td>
      <td>Mrs</td>
      <td>1968-06-30</td>
      <td>1002</td>
      <td>2016-06-24</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>307</td>
      <td>2016-06-23</td>
      <td>NaT</td>
      <td>68</td>
      <td>m</td>
      <td>purple</td>
      <td>3</td>
      <td>19.90</td>
      <td>30870</td>
      <td>Mrs</td>
      <td>1958-11-20</td>
      <td>1016</td>
      <td>2016-06-24</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>429</td>
      <td>2016-06-23</td>
      <td>2016-06-27</td>
      <td>405</td>
      <td>41</td>
      <td>green</td>
      <td>18</td>
      <td>79.90</td>
      <td>30892</td>
      <td>Mrs</td>
      <td>1966-05-10</td>
      <td>1002</td>
      <td>2016-06-24</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79919</th>
      <td>99640</td>
      <td>2016-09-10</td>
      <td>2016-09-13</td>
      <td>71</td>
      <td>7</td>
      <td>black</td>
      <td>21</td>
      <td>49.95</td>
      <td>48171</td>
      <td>Mrs</td>
      <td>1971-09-12</td>
      <td>1002</td>
      <td>2016-09-11</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79930</th>
      <td>99814</td>
      <td>2016-09-11</td>
      <td>2016-09-13</td>
      <td>166</td>
      <td>38</td>
      <td>ocher</td>
      <td>6</td>
      <td>39.90</td>
      <td>48212</td>
      <td>Mrs</td>
      <td>NaT</td>
      <td>1002</td>
      <td>2016-09-12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79937</th>
      <td>99895</td>
      <td>2016-09-11</td>
      <td>2016-09-13</td>
      <td>98</td>
      <td>l</td>
      <td>green</td>
      <td>28</td>
      <td>49.90</td>
      <td>48226</td>
      <td>Mrs</td>
      <td>1973-11-25</td>
      <td>1003</td>
      <td>2016-09-12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79940</th>
      <td>99942</td>
      <td>2016-09-11</td>
      <td>2016-09-12</td>
      <td>39</td>
      <td>41</td>
      <td>blue</td>
      <td>26</td>
      <td>89.90</td>
      <td>48232</td>
      <td>Mrs</td>
      <td>1941-10-24</td>
      <td>1007</td>
      <td>2016-09-12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79941</th>
      <td>99954</td>
      <td>2016-09-11</td>
      <td>NaT</td>
      <td>1498</td>
      <td>42</td>
      <td>green</td>
      <td>6</td>
      <td>59.90</td>
      <td>48234</td>
      <td>Mrs</td>
      <td>1962-10-02</td>
      <td>1007</td>
      <td>2016-09-12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16688 rows × 16 columns</p>
</div>




```python
#Data with flag for invalid delivery date
data[data['delivery_date']<data['order_date']]
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_title</th>
      <th>user_dob</th>
      <th>user_state</th>
      <th>user_reg_date</th>
      <th>return</th>
      <th>Invalid_User_Reg_Date_Flag</th>
      <th>Invalid_Delivery_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>312</th>
      <td>3141</td>
      <td>2016-06-25</td>
      <td>1994-12-31</td>
      <td>32</td>
      <td>l</td>
      <td>red</td>
      <td>3</td>
      <td>21.90</td>
      <td>598</td>
      <td>Mrs</td>
      <td>1970-05-08</td>
      <td>1003</td>
      <td>2015-02-17</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>431</th>
      <td>4377</td>
      <td>2016-06-26</td>
      <td>1994-12-31</td>
      <td>126</td>
      <td>6+</td>
      <td>red</td>
      <td>21</td>
      <td>39.95</td>
      <td>31734</td>
      <td>Mrs</td>
      <td>1965-06-15</td>
      <td>1008</td>
      <td>2016-02-16</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>498</th>
      <td>5244</td>
      <td>2016-06-27</td>
      <td>1994-12-31</td>
      <td>27</td>
      <td>5</td>
      <td>brown</td>
      <td>19</td>
      <td>39.90</td>
      <td>31858</td>
      <td>Mr</td>
      <td>NaT</td>
      <td>1008</td>
      <td>2016-06-28</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>609</th>
      <td>6348</td>
      <td>2016-06-27</td>
      <td>1994-12-31</td>
      <td>388</td>
      <td>xxl</td>
      <td>black</td>
      <td>3</td>
      <td>49.90</td>
      <td>32010</td>
      <td>Mrs</td>
      <td>1973-06-11</td>
      <td>1001</td>
      <td>2015-02-17</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>610</th>
      <td>6350</td>
      <td>2016-06-27</td>
      <td>1994-12-31</td>
      <td>195</td>
      <td>xxl</td>
      <td>curry</td>
      <td>46</td>
      <td>9.90</td>
      <td>32010</td>
      <td>Mrs</td>
      <td>1973-06-11</td>
      <td>1001</td>
      <td>2015-02-17</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79626</th>
      <td>96777</td>
      <td>2016-09-07</td>
      <td>1994-12-31</td>
      <td>1743</td>
      <td>xl</td>
      <td>black</td>
      <td>20</td>
      <td>79.90</td>
      <td>47631</td>
      <td>Mrs</td>
      <td>1967-11-30</td>
      <td>1008</td>
      <td>2016-09-08</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79727</th>
      <td>97731</td>
      <td>2016-09-09</td>
      <td>1994-12-31</td>
      <td>2058</td>
      <td>5+</td>
      <td>grey</td>
      <td>4</td>
      <td>180.00</td>
      <td>47791</td>
      <td>Family</td>
      <td>1988-01-08</td>
      <td>1009</td>
      <td>2016-09-10</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79851</th>
      <td>98939</td>
      <td>2016-09-10</td>
      <td>1994-12-31</td>
      <td>475</td>
      <td>45</td>
      <td>grey</td>
      <td>1</td>
      <td>99.90</td>
      <td>48031</td>
      <td>Mrs</td>
      <td>1982-11-08</td>
      <td>1006</td>
      <td>2016-09-11</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79863</th>
      <td>99042</td>
      <td>2016-09-10</td>
      <td>1994-12-31</td>
      <td>1670</td>
      <td>39</td>
      <td>grey</td>
      <td>1</td>
      <td>184.91</td>
      <td>2649</td>
      <td>Mrs</td>
      <td>1979-02-07</td>
      <td>1001</td>
      <td>2015-06-14</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79887</th>
      <td>99368</td>
      <td>2016-09-10</td>
      <td>1994-12-31</td>
      <td>2154</td>
      <td>42</td>
      <td>grey</td>
      <td>7</td>
      <td>19.90</td>
      <td>48120</td>
      <td>Mrs</td>
      <td>1986-02-22</td>
      <td>1008</td>
      <td>2016-09-11</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>846 rows × 16 columns</p>
</div>




```python
#Creating Flags for missing values of dates i.e for missing delivery date & user bob date
data['Missing_Delivery_Date']=data['delivery_date'].apply(lambda x: 1 if pd.isnull(x) else 0)
data['User_Dob_Missing_Flag']=data['user_dob'].apply(lambda x: 1 if pd.isnull(x) else 0)
```


```python
#Creating Derived Variables
#Days for delivery is a variable which tells days taken to deliver the order after order is placed
#Age Customer is a variable which tells the age of the customer when he/she is placing the order
#Day_Delivery is the day at which order is delivered
#Month_Delivery is the month in which order is delivered
#Year_Delivery is the year in which order is delivered
data['Days_For_Delivery']=np.nan
data['Age_Customer']=np.nan
data['Tenure_Customer_days']=np.nan
data['Day_Delivery']=np.nan
data['Month_Delivery']=np.nan
data['Year_Delivery']=np.nan

for i in range(0,len(data)):
    if ((data.at[i,'Invalid_Delivery_Date']==0) and (data.at[i,'Missing_Delivery_Date']==0)):
        data.at[i,'Days_For_Delivery']=(data.at[i,'delivery_date']-data.at[i,'order_date']).days
        data.at[i,'Day_Delivery']=data.at[i,'delivery_date'].day
        data.at[i,'Month_Delivery']=data.at[i,'delivery_date'].month
        data.at[i,'Year_Delivery']=data.at[i,'delivery_date'].year
    if ((data.at[i,'Invalid_User_Reg_Date_Flag']==0) and (data.at[i,'User_Dob_Missing_Flag']==0)):
        data.at[i,'Age_Customer']=data.at[i,'order_date'].year-data.at[i,'user_dob'].year
    if data.at[i,'Invalid_User_Reg_Date_Flag']==0:
        data.at[i,'Tenure_Customer_days']=(data.at[i,'order_date']-data.at[i,'user_reg_date']).days 
```


```python
#Missing Value Imputation for Derived Variables
data['Days_For_Delivery'].fillna(data['Days_For_Delivery'].median(),inplace=True)
data['Age_Customer'].fillna(data['Age_Customer'].mode()[0],inplace=True)
data['Tenure_Customer_days'].fillna(data['Tenure_Customer_days'].median(),inplace=True)
data['Day_Delivery'].fillna(data['Day_Delivery'].mode()[0],inplace=True)
data['Month_Delivery'].fillna(data['Month_Delivery'].mode()[0],inplace=True)
data['Year_Delivery'].fillna(data['Year_Delivery'].mode()[0],inplace=True)
```


```python
#Dummy Variables for Top 10 most occuring item id's i.e items which are ordered most 
top_10_item_id=list(data['item_id'].value_counts().to_frame().nlargest(10,'item_id').index)
data['item_id_new']=data['item_id'].apply(lambda x: x if x in top_10_item_id else 'other')
data=pd.get_dummies(data,columns=['item_id_new'],prefix='item_id').drop(['item_id_other'],axis=1)
```


```python
#Label encoding of Top 10 item sizes which had most returns
top10_item_size=list(data[data['return']==1]['item_size'].value_counts().to_frame().nlargest(10,'item_size').index)
top10_item_size_rank=data[data['return']==1]['item_size'].value_counts().to_frame().nlargest(10,'item_size').rank()
data['item_size_new']=data['item_size'].apply(lambda x: x if x in top10_item_size else 'other')
top10_item_size_rank=pd.concat([top10_item_size_rank,pd.DataFrame({'item_size':0},index=['other'])])
data['item_size_new']=data['item_size_new'].apply(lambda x:top10_item_size_rank.at[x,'item_size'])
```


```python
#Label Encoding of Top 10 items colors which had most returns
top10_item_color=list(data[data['return']==1]['item_color'].value_counts().to_frame().nlargest(10,'item_color').index)
top10_item_color_rank=data[data['return']==1]['item_color'].value_counts().to_frame().nlargest(10,'item_color').rank()
data['item_color_new']=data['item_color'].apply(lambda x: x if x in top10_item_color else 'other')
top10_item_color_rank=pd.concat([top10_item_color_rank,pd.DataFrame({'item_color':0},index=['other'])])
data['item_color_new']=data['item_color_new'].apply(lambda x:top10_item_color_rank.at[x,'item_color'])
```


```python
#Label Encoding of Top 10 brands which had most returns
top10_brand_id=list(data[data['return']==1]['brand_id'].value_counts().to_frame().nlargest(10,'brand_id').index)
top10_brand_id_rank=data[data['return']==1]['brand_id'].value_counts().to_frame().nlargest(10,'brand_id').rank()
data['brand_id_new']=data['brand_id'].apply(lambda x: x if x in top10_brand_id else 'other')
top10_brand_id_rank=pd.concat([top10_brand_id_rank,pd.DataFrame({'brand_id':0},index=['other'])])
data['brand_id_new']=data['brand_id_new'].apply(lambda x:top10_brand_id_rank.at[x,'brand_id'])
```


```python
#Ranking Users on basis of % of returns they have made with higher rank to higher percentages
df_user=data['user_id'].value_counts().to_frame().rename({'user_id':'No of Orders'},axis=1).join(data[data['return']==1]['user_id'].value_counts().to_frame().rename({'user_id':'no_of_returns'},axis=1),how='left')
df_user['no_of_returns'].fillna(0.0001,inplace=True)
df_user['%of returns']=(df_user['no_of_returns']/df_user['No of Orders'])*100
df_user=df_user['%of returns'].rank(method='dense').to_frame()
df_user=df_user.reset_index()
data=pd.merge(left=data,right=df_user,how='left',left_on='user_id',right_on='index').rename({'%of returns':'user_id_new'},axis=1)
data.drop(['index'],axis=1,inplace=True)
```


```python
#Saving User Ranking to use in scoring code
df_user.rename({'index':'user_id','%of returns':'user_id_new'},axis=1).to_csv(path+'Output_Scoring/user_id_ranking.csv',index=False)
```


```python
#Creating Dummy Variables for user title
data=pd.get_dummies(data,prefix='user_title',columns=['user_title'],drop_first=True)
```


```python
#Creating Dummy Variables for user state
data=pd.get_dummies(data,prefix='user_state',columns=['user_state'],drop_first=True)
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
      <th>order_item_id</th>
      <th>order_date</th>
      <th>delivery_date</th>
      <th>item_id</th>
      <th>item_size</th>
      <th>item_color</th>
      <th>brand_id</th>
      <th>item_price</th>
      <th>user_id</th>
      <th>user_dob</th>
      <th>user_reg_date</th>
      <th>return</th>
      <th>Invalid_User_Reg_Date_Flag</th>
      <th>Invalid_Delivery_Date</th>
      <th>Missing_Delivery_Date</th>
      <th>User_Dob_Missing_Flag</th>
      <th>Days_For_Delivery</th>
      <th>Age_Customer</th>
      <th>Tenure_Customer_days</th>
      <th>Day_Delivery</th>
      <th>Month_Delivery</th>
      <th>Year_Delivery</th>
      <th>item_id_22</th>
      <th>item_id_32</th>
      <th>item_id_100</th>
      <th>item_id_1401</th>
      <th>item_id_1415</th>
      <th>item_id_1445</th>
      <th>item_id_1470</th>
      <th>item_id_1532</th>
      <th>item_id_1546</th>
      <th>item_id_1607</th>
      <th>item_size_new</th>
      <th>item_color_new</th>
      <th>brand_id_new</th>
      <th>user_id_new</th>
      <th>user_title_Family</th>
      <th>user_title_Mr</th>
      <th>user_title_Mrs</th>
      <th>user_title_not reported</th>
      <th>user_state_1002</th>
      <th>user_state_1003</th>
      <th>user_state_1004</th>
      <th>user_state_1005</th>
      <th>user_state_1006</th>
      <th>user_state_1007</th>
      <th>user_state_1008</th>
      <th>user_state_1009</th>
      <th>user_state_1010</th>
      <th>user_state_1011</th>
      <th>user_state_1012</th>
      <th>user_state_1013</th>
      <th>user_state_1014</th>
      <th>user_state_1015</th>
      <th>user_state_1016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2016-06-22</td>
      <td>2016-06-27</td>
      <td>643</td>
      <td>38</td>
      <td>navy</td>
      <td>30</td>
      <td>49.9</td>
      <td>30822</td>
      <td>1969-04-17</td>
      <td>2016-06-23</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>50.0</td>
      <td>420.0</td>
      <td>27.0</td>
      <td>6.0</td>
      <td>2016.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>2016-06-22</td>
      <td>2016-06-27</td>
      <td>195</td>
      <td>xxl</td>
      <td>grey</td>
      <td>46</td>
      <td>19.9</td>
      <td>30823</td>
      <td>1970-04-22</td>
      <td>2015-03-15</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>46.0</td>
      <td>465.0</td>
      <td>27.0</td>
      <td>6.0</td>
      <td>2016.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>175.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>2016-06-22</td>
      <td>2016-07-05</td>
      <td>25</td>
      <td>xxl</td>
      <td>grey</td>
      <td>5</td>
      <td>79.9</td>
      <td>30823</td>
      <td>1970-04-22</td>
      <td>2015-03-15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0</td>
      <td>46.0</td>
      <td>465.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>2016.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>175.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>2016-06-23</td>
      <td>2016-06-26</td>
      <td>173</td>
      <td>m</td>
      <td>brown</td>
      <td>20</td>
      <td>19.9</td>
      <td>17234</td>
      <td>1960-01-09</td>
      <td>2015-02-17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>56.0</td>
      <td>492.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>2016.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>128.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43</td>
      <td>2016-06-23</td>
      <td>2016-06-26</td>
      <td>394</td>
      <td>40</td>
      <td>black</td>
      <td>44</td>
      <td>90.0</td>
      <td>30827</td>
      <td>NaT</td>
      <td>2016-02-09</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3.0</td>
      <td>50.0</td>
      <td>135.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>2016.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>263.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
</div>




```python
#Dropping Extra Features which are not required
data_features=data.copy()
data_features.drop(['order_item_id','order_date','delivery_date','item_id','item_size','item_color','brand_id','user_id','user_dob','user_reg_date'],axis=1,inplace=True)
```


```python
#Changing Data Types
data_features['Days_For_Delivery']=data_features['Days_For_Delivery'].astype(int)
data_features['Age_Customer']=data_features['Age_Customer'].astype(int)
data_features['Tenure_Customer_days']=data_features['Tenure_Customer_days'].astype(int)
data_features['Day_Delivery']=data_features['Day_Delivery'].astype(int)
data_features['Month_Delivery']=data_features['Month_Delivery'].astype(int)
data_features['Year_Delivery']=data_features['Year_Delivery'].astype(int)
data_features['item_size_new']=data_features['item_size_new'].astype(int)
data_features['item_color_new']=data_features['item_color_new'].astype(int)
data_features['brand_id_new']=data_features['brand_id_new'].astype(int)
data_features['user_id_new']=data_features['user_id_new'].astype(int)
```


```python
#Renaming Columns
data_features.rename({'item_size_new':'top_item_size_label','item_color_new':'top_item_color_label','brand_id_new':'top_brand_id_label','user_id_new':'user_id_label'},axis=1,inplace=True)
```


```python
data_features.head()
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
      <th>item_price</th>
      <th>return</th>
      <th>Invalid_User_Reg_Date_Flag</th>
      <th>Invalid_Delivery_Date</th>
      <th>Missing_Delivery_Date</th>
      <th>User_Dob_Missing_Flag</th>
      <th>Days_For_Delivery</th>
      <th>Age_Customer</th>
      <th>Tenure_Customer_days</th>
      <th>Day_Delivery</th>
      <th>Month_Delivery</th>
      <th>Year_Delivery</th>
      <th>item_id_22</th>
      <th>item_id_32</th>
      <th>item_id_100</th>
      <th>item_id_1401</th>
      <th>item_id_1415</th>
      <th>item_id_1445</th>
      <th>item_id_1470</th>
      <th>item_id_1532</th>
      <th>item_id_1546</th>
      <th>item_id_1607</th>
      <th>top_item_size_label</th>
      <th>top_item_color_label</th>
      <th>top_brand_id_label</th>
      <th>user_id_label</th>
      <th>user_title_Family</th>
      <th>user_title_Mr</th>
      <th>user_title_Mrs</th>
      <th>user_title_not reported</th>
      <th>user_state_1002</th>
      <th>user_state_1003</th>
      <th>user_state_1004</th>
      <th>user_state_1005</th>
      <th>user_state_1006</th>
      <th>user_state_1007</th>
      <th>user_state_1008</th>
      <th>user_state_1009</th>
      <th>user_state_1010</th>
      <th>user_state_1011</th>
      <th>user_state_1012</th>
      <th>user_state_1013</th>
      <th>user_state_1014</th>
      <th>user_state_1015</th>
      <th>user_state_1016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>420</td>
      <td>27</td>
      <td>6</td>
      <td>2016</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>46</td>
      <td>465</td>
      <td>27</td>
      <td>6</td>
      <td>2016</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>175</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79.9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>46</td>
      <td>465</td>
      <td>5</td>
      <td>7</td>
      <td>2016</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>6</td>
      <td>175</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>56</td>
      <td>492</td>
      <td>26</td>
      <td>6</td>
      <td>2016</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>8</td>
      <td>5</td>
      <td>128</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>90.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>50</td>
      <td>135</td>
      <td>26</td>
      <td>6</td>
      <td>2016</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>263</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
</div>



# Box Plots


```python
#Just plotting box plots to check distribution of numeric variables wont do outlier treatment since we will be using Random Forest Algorithm which is not sensitive to outliers
for col in ['item_price','Days_For_Delivery','Age_Customer','Month_Delivery']:
    sns.boxplot(y=data_features[col],x=data_features['return'])
    plt.show()
```


    
![png](https://ibb.co/VLNkJRc)
    



    
![png](https://ibb.co/KGZhxSY)
    



    
![png](https://ibb.co/CMVpwVX)
    



    
![png](https://ibb.co/JQSCfr4)
    


# Feature Selection


```python
#Checking For Imbalanced Classes , since return order are 46% of total order hence its not the case of imbalanced data.
(data_features['return'].value_counts()[1]/data_features.shape[0])*100
```




    45.859028081806244




```python
#Correlation Plot , non of the variables seems highly correlated with dependent or independent feature
sns.heatmap(data_features[['return','item_price','Days_For_Delivery','Age_Customer','Tenure_Customer_days','Day_Delivery','Month_Delivery','Year_Delivery','top_item_size_label','top_item_color_label','top_brand_id_label','user_id_label']].corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0b6ffdfdd0>




    
![png](https://ibb.co/5GcdxT4)
    



```python
#Creating Dependent and independent data
X=data_features.drop(['return'],axis=1)
Y=data_features[['return']].copy()
```


```python
#Splitting the data with 70-30 split and stratified sampling to get equal ratio of return class
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0,stratify=Y)
```


```python
#Using Recursive feature elimination with cross-validation to get most important features out of all features
cv_estimator = RandomForestClassifier(random_state =42,n_jobs=-1)
cv_estimator.fit(X_train, y_train)
cv_selector = RFECV(cv_estimator,cv= 5, step=1,scoring='accuracy')
cv_selector = cv_selector.fit(X_train, y_train)
```


```python
#Creating list of features which are most important
rfecv_mask=cv_selector.get_support()
rfecv_features = []
for bool, feature in zip(rfecv_mask, X_train.columns):
    if bool:
        rfecv_features.append(feature)
```


```python
#Most important features
rfecv_features
```




    ['item_price',
     'Missing_Delivery_Date',
     'Days_For_Delivery',
     'Age_Customer',
     'Tenure_Customer_days',
     'Day_Delivery',
     'Month_Delivery',
     'top_item_size_label',
     'top_item_color_label',
     'top_brand_id_label',
     'user_id_label',
     'user_state_1002',
     'user_state_1008',
     'user_state_1010']




```python
#Feature Importance Graph
n_features = X_train.shape[1]
plt.figure(figsize=(8,8))
plt.barh(range(n_features), cv_estimator.feature_importances_, align='center') 
plt.yticks(np.arange(n_features), X_train.columns.values) 
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
```


    
![png](https://ibb.co/1stNkHK)
    



```python
#Only picking best features
x_train_selected=X_train[rfecv_features]
x_test_selected=X_test[rfecv_features]
```

# Model Building with Hyperparameter Tunning - Using Random Forest due to time constraints , since random forest isn't sensitive to outliers and missing values . Also it doesn't require feature scaling.


```python
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
```


```python
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
```


```python
random_grid
```




    {'bootstrap': [True, False],
     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}




```python
#Calling Random Forest Classifier
rf=RandomForestClassifier()
```


```python
#Creating Randomized Search CV object
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
```


```python
#Fitting Random Forest Classifier
rf_random.fit(x_train_selected, y_train)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed: 20.8min
    [Parallel(n_jobs=-1)]: Done 158 tasks      | elapsed: 95.0min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 173.4min finished





    RandomizedSearchCV(cv=3, error_score=nan,
                       estimator=RandomForestClassifier(bootstrap=True,
                                                        ccp_alpha=0.0,
                                                        class_weight=None,
                                                        criterion='gini',
                                                        max_depth=None,
                                                        max_features='auto',
                                                        max_leaf_nodes=None,
                                                        max_samples=None,
                                                        min_impurity_decrease=0.0,
                                                        min_impurity_split=None,
                                                        min_samples_leaf=1,
                                                        min_samples_split=2,
                                                        min_weight_fraction_leaf=0.0,
                                                        n_estimators=100,
                                                        n_jobs...
                       param_distributions={'bootstrap': [True, False],
                                            'max_depth': [10, 20, 30, 40, 50, 60,
                                                          70, 80, 90, 100, 110,
                                                          None],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'n_estimators': [200, 400, 600, 800,
                                                             1000, 1200, 1400, 1600,
                                                             1800, 2000]},
                       pre_dispatch='2*n_jobs', random_state=42, refit=True,
                       return_train_score=False, scoring=None, verbose=2)




```python
#Best params obtained
rf_random.best_params_
```




    {'bootstrap': True,
     'max_depth': 10,
     'max_features': 'sqrt',
     'min_samples_leaf': 2,
     'min_samples_split': 5,
     'n_estimators': 1000}




```python
#Feature Importance Graph , here we could also remove user state dummy variables because they are not important
n_features = x_test_selected.shape[1]
plt.figure(figsize=(8,8))
plt.barh(range(n_features), rf_random.best_estimator_.feature_importances_, align='center') 
plt.yticks(np.arange(n_features), x_test_selected.columns.values) 
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
```


    
![png](https://ibb.co/pPpYXGn)
    


# Model Evaluation


```python
#Train Data Evaluation , using default 0.5 cutoff since in our case these are balanced classes

#Predicting Train classes
y_pred_train=rf_random.best_estimator_.predict(x_train_selected)

# Confusion Matrix
print("Confusion Matrix : \n",confusion_matrix(y_train, y_pred_train))
# Accuracy
print("Accuracy : \n",accuracy_score(y_train, y_pred_train))
# Recall
print("Recall : \n",recall_score(y_train, y_pred_train, average=None))
# Precision
print("Precision : \n",precision_score(y_train, y_pred_train, average=None))
# F1-Score
print("F1 Score : \n",f1_score(y_train, y_pred_train, average=None))
#AUC-ROC Curve
print("AUC-ROC Score : \n",roc_auc_score(y_train, y_pred_train, average=None))
#Classification Report
print(classification_report(y_train, y_pred_train))
```

    Confusion Matrix : 
     [[23249  7049]
     [ 3996 21667]]
    Accuracy : 
     0.8026304033165955
    Recall : 
     [0.76734438 0.84428944]
    Precision : 
     [0.85333089 0.75452709]
    F1 Score : 
     [0.80805658 0.7968885 ]
    AUC-ROC Score : 
     0.8058169115567397
                  precision    recall  f1-score   support
    
               0       0.85      0.77      0.81     30298
               1       0.75      0.84      0.80     25663
    
        accuracy                           0.80     55961
       macro avg       0.80      0.81      0.80     55961
    weighted avg       0.81      0.80      0.80     55961
    



```python
#Test Data Evaluation

#Predicting Train classes
y_pred_test=rf_random.best_estimator_.predict(x_test_selected)

# Confusion Matrix
print("Confusion Matrix : \n",confusion_matrix(y_test, y_pred_test))
# Accuracy
print("Accuracy : \n",accuracy_score(y_test, y_pred_test))
# Recall
print("Recall : \n",recall_score(y_test, y_pred_test, average=None))
# Precision
print("Precision : \n",precision_score(y_test, y_pred_test, average=None))
# F1-Score
print("F1 Score : \n",f1_score(y_test, y_pred_test, average=None))
#AUC-ROC Curve
print("AUC-ROC Score : \n",roc_auc_score(y_test, y_pred_test, average=None))
#Classification Report
print(classification_report(y_test, y_pred_test))
```

    Confusion Matrix : 
     [[9824 3161]
     [1874 9125]]
    Accuracy : 
     0.7900683789192795
    Recall : 
     [0.75656527 0.82962087]
    Precision : 
     [0.83980168 0.74271529]
    F1 Score : 
     [0.79601345 0.78376637]
    AUC-ROC Score : 
     0.7930930711207234
                  precision    recall  f1-score   support
    
               0       0.84      0.76      0.80     12985
               1       0.74      0.83      0.78     10999
    
        accuracy                           0.79     23984
       macro avg       0.79      0.79      0.79     23984
    weighted avg       0.80      0.79      0.79     23984
    


# Saving Model For Scoring


```python
filename = path+'Model/return_classifier.sav'
pickle.dump(rf_random.best_estimator_, open(filename, 'wb'))
```

