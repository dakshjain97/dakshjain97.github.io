---
layout: post
title: Kmeans clustering & profiling on wine data 
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [Kmeans, un-supervised learning, clusters evaluation, elbow method, silhouette_score]
author: Daksh Jain
---
Performed k-means clustering on wine data by choosing optimal clusters using elbow method & silhouette_scores , choosen clusters matched with oringal cluster/classes data had. Also at end using centroids profiling is done to compared means of clusters & actual classes.

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

    /kaggle/input/wine-pca/Wine.csv



```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
```


```python
df = pd.read_csv('/kaggle/input/wine-pca/Wine.csv')
```


```python
df.head()
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
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
      <th>Customer_Segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Customer_Segment'].value_counts()
```




    Customer_Segment
    2    71
    1    59
    3    48
    Name: count, dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 14 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Alcohol               178 non-null    float64
     1   Malic_Acid            178 non-null    float64
     2   Ash                   178 non-null    float64
     3   Ash_Alcanity          178 non-null    float64
     4   Magnesium             178 non-null    int64  
     5   Total_Phenols         178 non-null    float64
     6   Flavanoids            178 non-null    float64
     7   Nonflavanoid_Phenols  178 non-null    float64
     8   Proanthocyanins       178 non-null    float64
     9   Color_Intensity       178 non-null    float64
     10  Hue                   178 non-null    float64
     11  OD280                 178 non-null    float64
     12  Proline               178 non-null    int64  
     13  Customer_Segment      178 non-null    int64  
    dtypes: float64(11), int64(3)
    memory usage: 19.6 KB


# Data Exploration


```python
for col in df.columns:
    print(col)
    df[col].hist()
    plt.show()
```

    Alcohol



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_1.png)
    


    Malic_Acid



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_3.png)
    


    Ash



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_5.png)
    


    Ash_Alcanity



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_7.png)
    


    Magnesium



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_9.png)
    


    Total_Phenols



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_11.png)
    


    Flavanoids



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_13.png)
    


    Nonflavanoid_Phenols



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_15.png)
    


    Proanthocyanins



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_17.png)
    


    Color_Intensity



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_19.png)
    


    Hue



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_21.png)
    


    OD280



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_23.png)
    


    Proline



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_25.png)
    


    Customer_Segment



    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_7_27.png)
    



```python
#Correlation among variables
df_corr = df.corr()
df_corr[(df_corr>0.60) | (df_corr<-0.60)]
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
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
      <th>Customer_Segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alcohol</th>
      <td>1.00000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.643720</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Malic_Acid</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ash</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ash_Alcanity</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Magnesium</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Total_Phenols</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.864564</td>
      <td>NaN</td>
      <td>0.612413</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.699949</td>
      <td>NaN</td>
      <td>-0.719163</td>
    </tr>
    <tr>
      <th>Flavanoids</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.864564</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.652692</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.787194</td>
      <td>NaN</td>
      <td>-0.847498</td>
    </tr>
    <tr>
      <th>Nonflavanoid_Phenols</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Proanthocyanins</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.612413</td>
      <td>0.652692</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Color_Intensity</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Hue</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.617369</td>
    </tr>
    <tr>
      <th>OD280</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.699949</td>
      <td>0.787194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.788230</td>
    </tr>
    <tr>
      <th>Proline</th>
      <td>0.64372</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.633717</td>
    </tr>
    <tr>
      <th>Customer_Segment</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.719163</td>
      <td>-0.847498</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.617369</td>
      <td>-0.788230</td>
      <td>-0.633717</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Data Preperation


```python
cols = list(df.columns)
cols.remove('Customer_Segment')
```


```python
cols
```




    ['Alcohol',
     'Malic_Acid',
     'Ash',
     'Ash_Alcanity',
     'Magnesium',
     'Total_Phenols',
     'Flavanoids',
     'Nonflavanoid_Phenols',
     'Proanthocyanins',
     'Color_Intensity',
     'Hue',
     'OD280',
     'Proline']




```python
#Normalization
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]),columns = cols)

```


```python
df_scaled
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
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.518613</td>
      <td>-0.562250</td>
      <td>0.232053</td>
      <td>-1.169593</td>
      <td>1.913905</td>
      <td>0.808997</td>
      <td>1.034819</td>
      <td>-0.659563</td>
      <td>1.224884</td>
      <td>0.251717</td>
      <td>0.362177</td>
      <td>1.847920</td>
      <td>1.013009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.246290</td>
      <td>-0.499413</td>
      <td>-0.827996</td>
      <td>-2.490847</td>
      <td>0.018145</td>
      <td>0.568648</td>
      <td>0.733629</td>
      <td>-0.820719</td>
      <td>-0.544721</td>
      <td>-0.293321</td>
      <td>0.406051</td>
      <td>1.113449</td>
      <td>0.965242</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.196879</td>
      <td>0.021231</td>
      <td>1.109334</td>
      <td>-0.268738</td>
      <td>0.088358</td>
      <td>0.808997</td>
      <td>1.215533</td>
      <td>-0.498407</td>
      <td>2.135968</td>
      <td>0.269020</td>
      <td>0.318304</td>
      <td>0.788587</td>
      <td>1.395148</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.691550</td>
      <td>-0.346811</td>
      <td>0.487926</td>
      <td>-0.809251</td>
      <td>0.930918</td>
      <td>2.491446</td>
      <td>1.466525</td>
      <td>-0.981875</td>
      <td>1.032155</td>
      <td>1.186068</td>
      <td>-0.427544</td>
      <td>1.184071</td>
      <td>2.334574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.295700</td>
      <td>0.227694</td>
      <td>1.840403</td>
      <td>0.451946</td>
      <td>1.281985</td>
      <td>0.808997</td>
      <td>0.663351</td>
      <td>0.226796</td>
      <td>0.401404</td>
      <td>-0.319276</td>
      <td>0.362177</td>
      <td>0.449601</td>
      <td>-0.037874</td>
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
    </tr>
    <tr>
      <th>173</th>
      <td>0.876275</td>
      <td>2.974543</td>
      <td>0.305159</td>
      <td>0.301803</td>
      <td>-0.332922</td>
      <td>-0.985614</td>
      <td>-1.424900</td>
      <td>1.274310</td>
      <td>-0.930179</td>
      <td>1.142811</td>
      <td>-1.392758</td>
      <td>-1.231206</td>
      <td>-0.021952</td>
    </tr>
    <tr>
      <th>174</th>
      <td>0.493343</td>
      <td>1.412609</td>
      <td>0.414820</td>
      <td>1.052516</td>
      <td>0.158572</td>
      <td>-0.793334</td>
      <td>-1.284344</td>
      <td>0.549108</td>
      <td>-0.316950</td>
      <td>0.969783</td>
      <td>-1.129518</td>
      <td>-1.485445</td>
      <td>0.009893</td>
    </tr>
    <tr>
      <th>175</th>
      <td>0.332758</td>
      <td>1.744744</td>
      <td>-0.389355</td>
      <td>0.151661</td>
      <td>1.422412</td>
      <td>-1.129824</td>
      <td>-1.344582</td>
      <td>0.549108</td>
      <td>-0.422075</td>
      <td>2.224236</td>
      <td>-1.612125</td>
      <td>-1.485445</td>
      <td>0.280575</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0.209232</td>
      <td>0.227694</td>
      <td>0.012732</td>
      <td>0.151661</td>
      <td>1.422412</td>
      <td>-1.033684</td>
      <td>-1.354622</td>
      <td>1.354888</td>
      <td>-0.229346</td>
      <td>1.834923</td>
      <td>-1.568252</td>
      <td>-1.400699</td>
      <td>0.296498</td>
    </tr>
    <tr>
      <th>177</th>
      <td>1.395086</td>
      <td>1.583165</td>
      <td>1.365208</td>
      <td>1.502943</td>
      <td>-0.262708</td>
      <td>-0.392751</td>
      <td>-1.274305</td>
      <td>1.596623</td>
      <td>-0.422075</td>
      <td>1.791666</td>
      <td>-1.524378</td>
      <td>-1.428948</td>
      <td>-0.595160</td>
    </tr>
  </tbody>
</table>
<p>178 rows × 13 columns</p>
</div>



# K-means clustering

Since we already know that there are 3 clusters (from customer segment column) we can try to find what optimal cluster value does k means provide


```python
#check the optimal k value using elbow method
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k,n_init='auto')
    model.fit(df_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()
```


    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_16_0.png)
    


From above graph we can observe after k = 3 the decline in inertia in linear


```python
#check the optimal k value using silhouette_score
ks = range(2, 10)
scores = []

for k in ks:
    model = KMeans(n_clusters=k,n_init='auto')
    model.fit(df_scaled)
    scores.append(silhouette_score(df_scaled,model.labels_))

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, scores, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('silhouette_score')
plt.xticks(ks)
plt.show()
```


    
![png](https://dakshjain97.github.io/assets/img/kmean-clustering-wine-data-profiling_files/kmean-clustering-wine-data-profiling_18_0.png)
    


Using silhoute scores too at k = 3 has highest score values


```python
#Comparing the centroid with try means
k_means = KMeans(n_clusters = 3, random_state=123, n_init=30)
k_means.fit(df_scaled)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=3, n_init=30, random_state=123)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=3, n_init=30, random_state=123)</pre></div></div></div></div></div>




```python
pd.concat([df_scaled,df['Customer_Segment']],axis = 1).groupby(['Customer_Segment']).mean()
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
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
    <tr>
      <th>Customer_Segment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.919195</td>
      <td>-0.292342</td>
      <td>0.325604</td>
      <td>-0.737997</td>
      <td>0.463226</td>
      <td>0.873362</td>
      <td>0.956884</td>
      <td>-0.578985</td>
      <td>0.540383</td>
      <td>0.203401</td>
      <td>0.458847</td>
      <td>0.771351</td>
      <td>1.174501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.891720</td>
      <td>-0.362362</td>
      <td>-0.444958</td>
      <td>0.223137</td>
      <td>-0.364567</td>
      <td>-0.058067</td>
      <td>0.051780</td>
      <td>0.014569</td>
      <td>0.069002</td>
      <td>-0.852799</td>
      <td>0.433611</td>
      <td>0.245294</td>
      <td>-0.724110</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.189159</td>
      <td>0.895331</td>
      <td>0.257945</td>
      <td>0.577065</td>
      <td>-0.030127</td>
      <td>-0.987617</td>
      <td>-1.252761</td>
      <td>0.690119</td>
      <td>-0.766287</td>
      <td>1.011418</td>
      <td>-1.205382</td>
      <td>-1.310950</td>
      <td>-0.372578</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(k_means.cluster_centers_,columns = cols)
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
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.164907</td>
      <td>0.871547</td>
      <td>0.186898</td>
      <td>0.524367</td>
      <td>-0.075473</td>
      <td>-0.979330</td>
      <td>-1.215248</td>
      <td>0.726064</td>
      <td>-0.779706</td>
      <td>0.941539</td>
      <td>-1.164789</td>
      <td>-1.292412</td>
      <td>-0.407088</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.926072</td>
      <td>-0.394042</td>
      <td>-0.494517</td>
      <td>0.170602</td>
      <td>-0.491712</td>
      <td>-0.075983</td>
      <td>0.020813</td>
      <td>-0.033534</td>
      <td>0.058266</td>
      <td>-0.901914</td>
      <td>0.461804</td>
      <td>0.270764</td>
      <td>-0.753846</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.835232</td>
      <td>-0.303810</td>
      <td>0.364706</td>
      <td>-0.610191</td>
      <td>0.577587</td>
      <td>0.885237</td>
      <td>0.977820</td>
      <td>-0.562090</td>
      <td>0.580287</td>
      <td>0.171063</td>
      <td>0.473984</td>
      <td>0.779247</td>
      <td>1.125185</td>
    </tr>
  </tbody>
</table>
</div>



Here we observe cluster 
1. 0 centroid is close to customer_segment 3
2. 1 centroid is close to customer_segment 2
3. 2 centroid is close to customer_segment 1
