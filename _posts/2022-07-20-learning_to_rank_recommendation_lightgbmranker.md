---
layout: post
title: Anime Recommendation using learning to rank
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [recommendation, learning to rank, LightGBMRanker, NDCG, optuna]
author: Daksh Jain
---
This notebooks demonstrates a use case of anime recommendation ranking using learning to rank method, i.e for a user_id different anime titles are ranked on basis of predicted scores from LighGBMRanker & compared with true rankings ( for true rankings user ratings is taken as a proxy ). Hyper parameter tuning is done using Optuna & model is evaluated using NDCG metric.


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

    /kaggle/input/anime-recommendation-database-2020/watching_status.csv
    /kaggle/input/anime-recommendation-database-2020/rating_complete.csv
    /kaggle/input/anime-recommendation-database-2020/animelist.csv
    /kaggle/input/anime-recommendation-database-2020/anime.csv
    /kaggle/input/anime-recommendation-database-2020/anime_with_synopsis.csv
    /kaggle/input/anime-recommendation-database-2020/html folder/instructions.txt
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/reviews_1.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/staff.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/reviews_2.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/pictures.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/stats.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/details.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/reviews_3.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/5/recomendations.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_9.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_8.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_16.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_1.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_18.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/staff.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_4.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_11.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_19.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_2.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_13.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_17.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/pictures.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_22.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_10.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_15.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/stats.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_6.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/details.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_3.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_5.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_7.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_14.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_20.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_21.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/reviews_12.html
    /kaggle/input/anime-recommendation-database-2020/html folder/html/1/recomendations.html



```python
#Importing libraries
import gc
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
pd.set_option('display.max_columns', None)
```

# Data Cleaning & Data Exploration


```python
#Importing anime & rating data
anime = pd.read_csv('../input/anime-recommendation-database-2020/anime.csv')
rating = pd.read_csv('../input/anime-recommendation-database-2020/rating_complete.csv')
```


```python
#This anime data stores details of different anime titles ~17K animes along with certain features
anime.head()
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Source</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Ranked</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Watching</th>
      <th>Completed</th>
      <th>On-Hold</th>
      <th>Dropped</th>
      <th>Plan to Watch</th>
      <th>Score-10</th>
      <th>Score-9</th>
      <th>Score-8</th>
      <th>Score-7</th>
      <th>Score-6</th>
      <th>Score-5</th>
      <th>Score-4</th>
      <th>Score-3</th>
      <th>Score-2</th>
      <th>Score-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>TV</td>
      <td>26</td>
      <td>Apr 3, 1998 to Apr 24, 1999</td>
      <td>Spring 1998</td>
      <td>Bandai Visual</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>Sunrise</td>
      <td>Original</td>
      <td>24 min. per ep.</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>28.0</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>329800</td>
      <td>229170.0</td>
      <td>182126.0</td>
      <td>131625.0</td>
      <td>62330.0</td>
      <td>20688.0</td>
      <td>8904.0</td>
      <td>3184.0</td>
      <td>1357.0</td>
      <td>741.0</td>
      <td>1580.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>8.39</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>Cowboy Bebop:The Movie</td>
      <td>カウボーイビバップ 天国の扉</td>
      <td>Movie</td>
      <td>1</td>
      <td>Sep 1, 2001</td>
      <td>Unknown</td>
      <td>Sunrise, Bandai Visual</td>
      <td>Sony Pictures Entertainment</td>
      <td>Bones</td>
      <td>Original</td>
      <td>1 hr. 55 min.</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>159.0</td>
      <td>518</td>
      <td>273145</td>
      <td>1174</td>
      <td>4143</td>
      <td>208333</td>
      <td>1935</td>
      <td>770</td>
      <td>57964</td>
      <td>30043.0</td>
      <td>49201.0</td>
      <td>49505.0</td>
      <td>22632.0</td>
      <td>5805.0</td>
      <td>1877.0</td>
      <td>577.0</td>
      <td>221.0</td>
      <td>109.0</td>
      <td>379.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Trigun</td>
      <td>8.24</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>Trigun</td>
      <td>トライガン</td>
      <td>TV</td>
      <td>26</td>
      <td>Apr 1, 1998 to Sep 30, 1998</td>
      <td>Spring 1998</td>
      <td>Victor Entertainment</td>
      <td>Funimation, Geneon Entertainment USA</td>
      <td>Madhouse</td>
      <td>Manga</td>
      <td>24 min. per ep.</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>266.0</td>
      <td>201</td>
      <td>558913</td>
      <td>12944</td>
      <td>29113</td>
      <td>343492</td>
      <td>25465</td>
      <td>13925</td>
      <td>146918</td>
      <td>50229.0</td>
      <td>75651.0</td>
      <td>86142.0</td>
      <td>49432.0</td>
      <td>15376.0</td>
      <td>5838.0</td>
      <td>1965.0</td>
      <td>664.0</td>
      <td>316.0</td>
      <td>533.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Witch Hunter Robin</td>
      <td>7.27</td>
      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>
      <td>Witch Hunter Robin</td>
      <td>Witch Hunter ROBIN (ウイッチハンターロビン)</td>
      <td>TV</td>
      <td>26</td>
      <td>Jul 2, 2002 to Dec 24, 2002</td>
      <td>Summer 2002</td>
      <td>TV Tokyo, Bandai Visual, Dentsu, Victor Entert...</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>Sunrise</td>
      <td>Original</td>
      <td>25 min. per ep.</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>2481.0</td>
      <td>1467</td>
      <td>94683</td>
      <td>587</td>
      <td>4300</td>
      <td>46165</td>
      <td>5121</td>
      <td>5378</td>
      <td>33719</td>
      <td>2182.0</td>
      <td>4806.0</td>
      <td>10128.0</td>
      <td>11618.0</td>
      <td>5709.0</td>
      <td>2920.0</td>
      <td>1083.0</td>
      <td>353.0</td>
      <td>164.0</td>
      <td>131.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Bouken Ou Beet</td>
      <td>6.98</td>
      <td>Adventure, Fantasy, Shounen, Supernatural</td>
      <td>Beet the Vandel Buster</td>
      <td>冒険王ビィト</td>
      <td>TV</td>
      <td>52</td>
      <td>Sep 30, 2004 to Sep 29, 2005</td>
      <td>Fall 2004</td>
      <td>TV Tokyo, Dentsu</td>
      <td>Unknown</td>
      <td>Toei Animation</td>
      <td>Manga</td>
      <td>23 min. per ep.</td>
      <td>PG - Children</td>
      <td>3710.0</td>
      <td>4369</td>
      <td>13224</td>
      <td>18</td>
      <td>642</td>
      <td>7314</td>
      <td>766</td>
      <td>1108</td>
      <td>3394</td>
      <td>312.0</td>
      <td>529.0</td>
      <td>1242.0</td>
      <td>1713.0</td>
      <td>1068.0</td>
      <td>634.0</td>
      <td>265.0</td>
      <td>83.0</td>
      <td>50.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
anime.shape
```




    (17562, 35)




```python
#checking for duplicated values
anime[anime.duplicated()]
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Source</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Ranked</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Watching</th>
      <th>Completed</th>
      <th>On-Hold</th>
      <th>Dropped</th>
      <th>Plan to Watch</th>
      <th>Score-10</th>
      <th>Score-9</th>
      <th>Score-8</th>
      <th>Score-7</th>
      <th>Score-6</th>
      <th>Score-5</th>
      <th>Score-4</th>
      <th>Score-3</th>
      <th>Score-2</th>
      <th>Score-1</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
anime.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17562 entries, 0 to 17561
    Data columns (total 35 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   MAL_ID         17562 non-null  int64 
     1   Name           17562 non-null  object
     2   Score          17562 non-null  object
     3   Genres         17562 non-null  object
     4   English name   17562 non-null  object
     5   Japanese name  17562 non-null  object
     6   Type           17562 non-null  object
     7   Episodes       17562 non-null  object
     8   Aired          17562 non-null  object
     9   Premiered      17562 non-null  object
     10  Producers      17562 non-null  object
     11  Licensors      17562 non-null  object
     12  Studios        17562 non-null  object
     13  Source         17562 non-null  object
     14  Duration       17562 non-null  object
     15  Rating         17562 non-null  object
     16  Ranked         17562 non-null  object
     17  Popularity     17562 non-null  int64 
     18  Members        17562 non-null  int64 
     19  Favorites      17562 non-null  int64 
     20  Watching       17562 non-null  int64 
     21  Completed      17562 non-null  int64 
     22  On-Hold        17562 non-null  int64 
     23  Dropped        17562 non-null  int64 
     24  Plan to Watch  17562 non-null  int64 
     25  Score-10       17562 non-null  object
     26  Score-9        17562 non-null  object
     27  Score-8        17562 non-null  object
     28  Score-7        17562 non-null  object
     29  Score-6        17562 non-null  object
     30  Score-5        17562 non-null  object
     31  Score-4        17562 non-null  object
     32  Score-3        17562 non-null  object
     33  Score-2        17562 non-null  object
     34  Score-1        17562 non-null  object
    dtypes: int64(9), object(26)
    memory usage: 4.7+ MB



```python
rating.head()
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
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>430</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1004</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3010</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>570</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2762</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating.shape
```




    (57633278, 3)




```python
#example ratings listed by a user across different anime titles
rating[rating['user_id']==353404]
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
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57633248</th>
      <td>353404</td>
      <td>897</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633249</th>
      <td>353404</td>
      <td>223</td>
      <td>9</td>
    </tr>
    <tr>
      <th>57633250</th>
      <td>353404</td>
      <td>898</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633251</th>
      <td>353404</td>
      <td>899</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633252</th>
      <td>353404</td>
      <td>900</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633253</th>
      <td>353404</td>
      <td>901</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633254</th>
      <td>353404</td>
      <td>902</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633255</th>
      <td>353404</td>
      <td>903</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633256</th>
      <td>353404</td>
      <td>904</td>
      <td>6</td>
    </tr>
    <tr>
      <th>57633257</th>
      <td>353404</td>
      <td>905</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633258</th>
      <td>353404</td>
      <td>906</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633259</th>
      <td>353404</td>
      <td>986</td>
      <td>9</td>
    </tr>
    <tr>
      <th>57633260</th>
      <td>353404</td>
      <td>985</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633261</th>
      <td>353404</td>
      <td>287</td>
      <td>9</td>
    </tr>
    <tr>
      <th>57633262</th>
      <td>353404</td>
      <td>895</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633263</th>
      <td>353404</td>
      <td>551</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633264</th>
      <td>353404</td>
      <td>507</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633265</th>
      <td>353404</td>
      <td>392</td>
      <td>9</td>
    </tr>
    <tr>
      <th>57633266</th>
      <td>353404</td>
      <td>882</td>
      <td>6</td>
    </tr>
    <tr>
      <th>57633267</th>
      <td>353404</td>
      <td>883</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633268</th>
      <td>353404</td>
      <td>894</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633269</th>
      <td>353404</td>
      <td>813</td>
      <td>9</td>
    </tr>
    <tr>
      <th>57633270</th>
      <td>353404</td>
      <td>893</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633271</th>
      <td>353404</td>
      <td>892</td>
      <td>6</td>
    </tr>
    <tr>
      <th>57633272</th>
      <td>353404</td>
      <td>891</td>
      <td>5</td>
    </tr>
    <tr>
      <th>57633273</th>
      <td>353404</td>
      <td>502</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633274</th>
      <td>353404</td>
      <td>987</td>
      <td>4</td>
    </tr>
    <tr>
      <th>57633275</th>
      <td>353404</td>
      <td>225</td>
      <td>8</td>
    </tr>
    <tr>
      <th>57633276</th>
      <td>353404</td>
      <td>243</td>
      <td>7</td>
    </tr>
    <tr>
      <th>57633277</th>
      <td>353404</td>
      <td>896</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sampling 10% of rating data for POC purpose
rating = rating.sample(frac = 0.1,random_state = 100)
rating.reset_index(drop = True, inplace = True)
```


```python
rating.shape
```




    (5763328, 3)




```python
rating.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5763328 entries, 0 to 5763327
    Data columns (total 3 columns):
     #   Column    Dtype
    ---  ------    -----
     0   user_id   int64
     1   anime_id  int64
     2   rating    int64
    dtypes: int64(3)
    memory usage: 131.9 MB



```python
#Unique anime titles
anime['MAL_ID'].nunique()
```




    17562




```python
#unique users who listed ratings
rating['user_id'].nunique()
```




    285794




```python
# use anime features
anime_features = ['MAL_ID','English name','Japanese name','Score','Genres','Popularity','Members',
            'Favorites','Watching','Completed','On-Hold','Dropped',
            'Score-1','Score-2','Score-3','Score-4','Score-5',
            'Score-6','Score-7','Score-8','Score-9','Score-10',
           ]
anime = anime[anime_features]
```


```python
anime.head()
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
      <th>MAL_ID</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Watching</th>
      <th>Completed</th>
      <th>On-Hold</th>
      <th>Dropped</th>
      <th>Score-1</th>
      <th>Score-2</th>
      <th>Score-3</th>
      <th>Score-4</th>
      <th>Score-5</th>
      <th>Score-6</th>
      <th>Score-7</th>
      <th>Score-8</th>
      <th>Score-9</th>
      <th>Score-10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Cowboy Bebop:The Movie</td>
      <td>カウボーイビバップ 天国の扉</td>
      <td>8.39</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>518</td>
      <td>273145</td>
      <td>1174</td>
      <td>4143</td>
      <td>208333</td>
      <td>1935</td>
      <td>770</td>
      <td>379.0</td>
      <td>109.0</td>
      <td>221.0</td>
      <td>577.0</td>
      <td>1877.0</td>
      <td>5805.0</td>
      <td>22632.0</td>
      <td>49505.0</td>
      <td>49201.0</td>
      <td>30043.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Trigun</td>
      <td>トライガン</td>
      <td>8.24</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>201</td>
      <td>558913</td>
      <td>12944</td>
      <td>29113</td>
      <td>343492</td>
      <td>25465</td>
      <td>13925</td>
      <td>533.0</td>
      <td>316.0</td>
      <td>664.0</td>
      <td>1965.0</td>
      <td>5838.0</td>
      <td>15376.0</td>
      <td>49432.0</td>
      <td>86142.0</td>
      <td>75651.0</td>
      <td>50229.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Witch Hunter Robin</td>
      <td>Witch Hunter ROBIN (ウイッチハンターロビン)</td>
      <td>7.27</td>
      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>
      <td>1467</td>
      <td>94683</td>
      <td>587</td>
      <td>4300</td>
      <td>46165</td>
      <td>5121</td>
      <td>5378</td>
      <td>131.0</td>
      <td>164.0</td>
      <td>353.0</td>
      <td>1083.0</td>
      <td>2920.0</td>
      <td>5709.0</td>
      <td>11618.0</td>
      <td>10128.0</td>
      <td>4806.0</td>
      <td>2182.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Beet the Vandel Buster</td>
      <td>冒険王ビィト</td>
      <td>6.98</td>
      <td>Adventure, Fantasy, Shounen, Supernatural</td>
      <td>4369</td>
      <td>13224</td>
      <td>18</td>
      <td>642</td>
      <td>7314</td>
      <td>766</td>
      <td>1108</td>
      <td>27.0</td>
      <td>50.0</td>
      <td>83.0</td>
      <td>265.0</td>
      <td>634.0</td>
      <td>1068.0</td>
      <td>1713.0</td>
      <td>1242.0</td>
      <td>529.0</td>
      <td>312.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating.shape
```




    (5763328, 3)




```python
# merge anime and rating to get ratings of a user across different anime titles
merged_df = anime.merge(rating, left_on='MAL_ID', right_on='anime_id', how='inner')
print(merged_df.shape)
merged_df.head()
```

    (5763328, 25)





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
      <th>MAL_ID</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Watching</th>
      <th>Completed</th>
      <th>On-Hold</th>
      <th>Dropped</th>
      <th>Score-1</th>
      <th>Score-2</th>
      <th>Score-3</th>
      <th>Score-4</th>
      <th>Score-5</th>
      <th>Score-6</th>
      <th>Score-7</th>
      <th>Score-8</th>
      <th>Score-9</th>
      <th>Score-10</th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>332241</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>133482</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>231084</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>263655</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>246289</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



# Feature creation


```python
# use genres
genre_names = [
    'Action', 'Adventure','Comedy',
    'Slice of Life','Drama','Sci-Fi',
    'Game','Harem','Military','Space','Music', 'Mecha',
     'Historical', 'Mystery', 'School', 'Hentai', 'Fantasy', 'Horror',
     'Kids', 'Sports', 'Magic', 'Romance', 
]

#Function to one-hot encode genres
def genre_to_category(df):
    '''Add genre cagegory column
    '''
    d = {name :[] for name in genre_names}
    
    def f(row):
        genres = row.Genres.split(',')
        genres = [gn.strip() for gn in genres]
        for genre in genre_names:
            if genre in genres:
                d[genre].append(1)
            else:
                d[genre].append(0)

    # create genre category dict
    df.apply(f, axis=1)
    
    # add genre category
    genre_df = pd.DataFrame(d, columns=genre_names)
    df = pd.concat([df, genre_df], axis=1)
    return df

#Function to clean score feature and create genre feature
def make_anime_feature(df):
    # convert object to a numeric type, replacing Unknown with nan.
    df['Score'] = df['Score'].apply(lambda x: np.nan if x=='Unknown' else float(x)) 
    for i in range(1, 11):
        df[f'Score-{i}'] = df[f'Score-{i}'].apply(lambda x: np.nan if x=='Unknown' else float(x))
    
    # add genre ctegory columns
    df = genre_to_category(df)
    
    return df

#function to create features at user level (across all anime titles)
def make_user_feature(df):
    # add user feature
    df['rating_count'] = df.groupby('user_id')['anime_id'].transform('count')
    df['rating_mean'] = df.groupby('user_id')['rating'].transform('mean')
    return df

#Function to preprocess and create features
def preprocess(merged_df):
    merged_df = make_anime_feature(merged_df)
    merged_df = make_user_feature(merged_df)
    return merged_df
```


```python
merged_df = preprocess(merged_df)
merged_df.head()
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
      <th>MAL_ID</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Watching</th>
      <th>Completed</th>
      <th>On-Hold</th>
      <th>Dropped</th>
      <th>Score-1</th>
      <th>Score-2</th>
      <th>Score-3</th>
      <th>Score-4</th>
      <th>Score-5</th>
      <th>Score-6</th>
      <th>Score-7</th>
      <th>Score-8</th>
      <th>Score-9</th>
      <th>Score-10</th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Comedy</th>
      <th>Slice of Life</th>
      <th>Drama</th>
      <th>Sci-Fi</th>
      <th>Game</th>
      <th>Harem</th>
      <th>Military</th>
      <th>Space</th>
      <th>Music</th>
      <th>Mecha</th>
      <th>Historical</th>
      <th>Mystery</th>
      <th>School</th>
      <th>Hentai</th>
      <th>Fantasy</th>
      <th>Horror</th>
      <th>Kids</th>
      <th>Sports</th>
      <th>Magic</th>
      <th>Romance</th>
      <th>rating_count</th>
      <th>rating_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>332241</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>7.250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>133482</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>8.130435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>231084</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>263655</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>7.642857</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>カウボーイビバップ</td>
      <td>8.78</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>39</td>
      <td>1251960</td>
      <td>61971</td>
      <td>105808</td>
      <td>718161</td>
      <td>71513</td>
      <td>26678</td>
      <td>1580.0</td>
      <td>741.0</td>
      <td>1357.0</td>
      <td>3184.0</td>
      <td>8904.0</td>
      <td>20688.0</td>
      <td>62330.0</td>
      <td>131625.0</td>
      <td>182126.0</td>
      <td>229170.0</td>
      <td>246289</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>7.857143</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_df.shape
```




    (5763328, 49)



# Model Training





```python
# random split
train, test = train_test_split(merged_df, test_size=0.2, random_state=100)
del merged_df
gc.collect()
```




    22543




```python
print('train shape: ',train.shape)
print('test shape: ',test.shape)
```

    train shape:  (4610662, 49)
    test shape:  (1152666, 49)



```python
features = ['Score', 'Popularity','Members',
            'Favorites','Watching','Completed','On-Hold','Dropped',
            'Score-1','Score-2','Score-3','Score-4','Score-5',
            'Score-6','Score-7','Score-8','Score-9','Score-10',
            'rating_count','rating_mean'
           ]
features += genre_names
user_col = 'user_id'
item_col = 'anime_id'
target_col = 'rating'
```


```python
train = train.sort_values('user_id').reset_index(drop=True)
test = test.sort_values('user_id').reset_index(drop=True)
```


```python
train.head()
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
      <th>MAL_ID</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Watching</th>
      <th>Completed</th>
      <th>On-Hold</th>
      <th>Dropped</th>
      <th>Score-1</th>
      <th>Score-2</th>
      <th>Score-3</th>
      <th>Score-4</th>
      <th>Score-5</th>
      <th>Score-6</th>
      <th>Score-7</th>
      <th>Score-8</th>
      <th>Score-9</th>
      <th>Score-10</th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Comedy</th>
      <th>Slice of Life</th>
      <th>Drama</th>
      <th>Sci-Fi</th>
      <th>Game</th>
      <th>Harem</th>
      <th>Military</th>
      <th>Space</th>
      <th>Music</th>
      <th>Mecha</th>
      <th>Historical</th>
      <th>Mystery</th>
      <th>School</th>
      <th>Hentai</th>
      <th>Fantasy</th>
      <th>Horror</th>
      <th>Kids</th>
      <th>Sports</th>
      <th>Magic</th>
      <th>Romance</th>
      <th>rating_count</th>
      <th>rating_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34134</td>
      <td>Unknown</td>
      <td>ワンパンマン</td>
      <td>7.41</td>
      <td>Action, Sci-Fi, Comedy, Parody, Super Power, S...</td>
      <td>70</td>
      <td>994488</td>
      <td>5937</td>
      <td>61959</td>
      <td>666395</td>
      <td>12925</td>
      <td>19642</td>
      <td>1654.0</td>
      <td>1758.0</td>
      <td>4305.0</td>
      <td>12969.0</td>
      <td>27047.0</td>
      <td>71761.0</td>
      <td>156260.0</td>
      <td>143291.0</td>
      <td>72761.0</td>
      <td>44526.0</td>
      <td>1</td>
      <td>34134</td>
      <td>7</td>
      <td>1</td>
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
      <td>9</td>
      <td>7.888889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37779</td>
      <td>The Promised Neverland</td>
      <td>約束のネバーランド</td>
      <td>8.65</td>
      <td>Sci-Fi, Mystery, Horror, Psychological, Thrill...</td>
      <td>55</td>
      <td>1133952</td>
      <td>32542</td>
      <td>78900</td>
      <td>863136</td>
      <td>17950</td>
      <td>12852</td>
      <td>1041.0</td>
      <td>373.0</td>
      <td>790.0</td>
      <td>2055.0</td>
      <td>5198.0</td>
      <td>15825.0</td>
      <td>66705.0</td>
      <td>202613.0</td>
      <td>258096.0</td>
      <td>169589.0</td>
      <td>1</td>
      <td>37779</td>
      <td>9</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>7.888889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9919</td>
      <td>Blue Exorcist</td>
      <td>青の祓魔師(エクソシスト)</td>
      <td>7.59</td>
      <td>Action, Demons, Fantasy, Shounen, Supernatural</td>
      <td>25</td>
      <td>1417630</td>
      <td>16871</td>
      <td>60451</td>
      <td>1105894</td>
      <td>36030</td>
      <td>38705</td>
      <td>1448.0</td>
      <td>1786.0</td>
      <td>4567.0</td>
      <td>13431.0</td>
      <td>40731.0</td>
      <td>99008.0</td>
      <td>240338.0</td>
      <td>248970.0</td>
      <td>137344.0</td>
      <td>80157.0</td>
      <td>1</td>
      <td>9919</td>
      <td>8</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>7.888889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31964</td>
      <td>My Hero Academia</td>
      <td>僕のヒーローアカデミア</td>
      <td>8.11</td>
      <td>Action, Comedy, School, Shounen, Super Power</td>
      <td>6</td>
      <td>1909814</td>
      <td>50005</td>
      <td>90902</td>
      <td>1655900</td>
      <td>18092</td>
      <td>19212</td>
      <td>3129.0</td>
      <td>1807.0</td>
      <td>3664.0</td>
      <td>9015.0</td>
      <td>29893.0</td>
      <td>77961.0</td>
      <td>253871.0</td>
      <td>414913.0</td>
      <td>318675.0</td>
      <td>192539.0</td>
      <td>1</td>
      <td>31964</td>
      <td>9</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>7.888889</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>Monster</td>
      <td>モンスター</td>
      <td>8.76</td>
      <td>Drama, Horror, Mystery, Police, Psychological,...</td>
      <td>169</td>
      <td>614100</td>
      <td>29436</td>
      <td>64648</td>
      <td>214491</td>
      <td>47488</td>
      <td>23008</td>
      <td>1177.0</td>
      <td>593.0</td>
      <td>882.0</td>
      <td>2086.0</td>
      <td>4381.0</td>
      <td>8861.0</td>
      <td>22045.0</td>
      <td>43459.0</td>
      <td>60652.0</td>
      <td>77350.0</td>
      <td>1</td>
      <td>19</td>
      <td>9</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>7.888889</td>
    </tr>
  </tbody>
</table>
</div>




```python
# model query data
train_query = train[user_col].value_counts().sort_index()
test_query = test[user_col].value_counts().sort_index()
```


```python
train_query.head()
```




    user_id
    1     9
    2     3
    3    35
    4    10
    5     6
    Name: count, dtype: int64




```python
# try parameter tuning
def objective(trial):
    # search param
    param = {
        'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1), 
        #'subsample': trial.suggest_uniform('subsample', 1e-8, 1), 
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 
    }
     
    #train model
    model = lgb.LGBMRanker(n_estimators=1000, **param, random_state=100, early_stopping_rounds=50,
        verbose=10)
    model.fit(
        train[features],
        train[target_col],
        group=train_query,
        eval_set=[(test[features], test[target_col])],
        eval_group=[list(test_query)],
        eval_at=[1, 3, 5, 10, 20], # calc validation ndcg@1,3,5,10,20
    )
    
    # maximize mean ndcg
    scores = []
    for name, score in model.best_score_['valid_0'].items():
        scores.append(score)
    return np.mean(scores)
 
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=100) #fix random seed
                           )
study.optimize(objective, n_trials=10)

Interupting training for demo purpose ( to limit notebook run time)

```

    [I 2024-03-08 10:23:17,710] A new study created in memory with name: no-name-0c1504ee-8b47-44ef-8d2f-61e141a46cb8
    /tmp/ipykernel_33/823259258.py:5: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    /tmp/ipykernel_33/823259258.py:6: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    /tmp/ipykernel_33/823259258.py:9: FutureWarning: suggest_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float instead.
      'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),


    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Debug] Dataset::GetMultiBinFromSparseFeatures: sparse rate 0.883502
    [LightGBM] [Debug] Dataset::GetMultiBinFromAllFeatures: sparse rate 0.447191
    [LightGBM] [Debug] init for col-wise cost 0.653255 seconds, init for row-wise cost 2.375380 seconds
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.754101 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Debug] Using Sparse Multi-Val Bin
    [LightGBM] [Info] Total Bins 5102
    [LightGBM] [Info] Number of data points in the train set: 4610662, number of used features: 42
    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    Training until validation scores don't improve for 50 rounds
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 24 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 19 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 24 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 12 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 17 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 20 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 10 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 9 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 12 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5


    [I 2024-03-08 10:27:49,920] Trial 0 finished with value: 0.901284958970684 and parameters: {'lambda_l1': 0.0007773998922821829, 'lambda_l2': 3.2012859298995277e-06, 'max_depth': 5, 'num_leaves': 217, 'colsample_bytree': 0.10424697057187532, 'min_child_samples': 16}. Best is trial 0 with value: 0.901284958970684.


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    Early stopping, best iteration is:
    [181]	valid_0's ndcg@1: 0.828856	valid_0's ndcg@3: 0.891943	valid_0's ndcg@5: 0.915653	valid_0's ndcg@10: 0.932575	valid_0's ndcg@20: 0.937398


    /tmp/ipykernel_33/823259258.py:5: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    /tmp/ipykernel_33/823259258.py:6: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    /tmp/ipykernel_33/823259258.py:9: FutureWarning: suggest_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float instead.
      'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),


    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Debug] Dataset::GetMultiBinFromSparseFeatures: sparse rate 0.883502
    [LightGBM] [Debug] Dataset::GetMultiBinFromAllFeatures: sparse rate 0.447191
    [LightGBM] [Debug] init for col-wise cost 1.039986 seconds, init for row-wise cost 2.672119 seconds
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 2.160673 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Debug] Using Sparse Multi-Val Bin
    [LightGBM] [Info] Total Bins 5102
    [LightGBM] [Info] Number of data points in the train set: 4610662, number of used features: 42
    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    Training until validation scores don't improve for 50 rounds
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3


    [I 2024-03-08 10:33:31,515] Trial 1 finished with value: 0.9008469186434352 and parameters: {'lambda_l1': 0.010882827930218712, 'lambda_l2': 0.2708162972907513, 'max_depth': 3, 'num_leaves': 148, 'colsample_bytree': 0.9021897588810376, 'min_child_samples': 25}. Best is trial 0 with value: 0.901284958970684.


    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    Early stopping, best iteration is:
    [84]	valid_0's ndcg@1: 0.828143	valid_0's ndcg@3: 0.891426	valid_0's ndcg@5: 0.91528	valid_0's ndcg@10: 0.932272	valid_0's ndcg@20: 0.937113


    /tmp/ipykernel_33/823259258.py:5: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    /tmp/ipykernel_33/823259258.py:6: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
      'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    /tmp/ipykernel_33/823259258.py:9: FutureWarning: suggest_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float instead.
      'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),


    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Debug] Dataset::GetMultiBinFromSparseFeatures: sparse rate 0.883502
    [LightGBM] [Debug] Dataset::GetMultiBinFromAllFeatures: sparse rate 0.447191
    [LightGBM] [Debug] init for col-wise cost 1.029835 seconds, init for row-wise cost 3.151893 seconds
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 4.205481 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 5102
    [LightGBM] [Info] Number of data points in the train set: 4610662, number of used features: 42
    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    Training until validation scores don't improve for 50 rounds
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 11 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 12 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 15 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4


    Exception ignored on calling ctypes callback function: <function _log_callback at 0x7b84d5655c60>
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.10/site-packages/lightgbm/basic.py", line 224, in _log_callback
        def _log_callback(msg: bytes) -> None:
    KeyboardInterrupt: 


    Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4


    [W 2024-03-08 10:51:49,580] Trial 2 failed with parameters: {'lambda_l1': 4.655367559816141e-07, 'lambda_l2': 9.449134137745608e-08, 'max_depth': 4, 'num_leaves': 251, 'colsample_bytree': 0.830514834180391, 'min_child_samples': 21} because of the following error: KeyboardInterrupt().
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.10/site-packages/optuna/study/_optimize.py", line 200, in _run_trial
        value_or_values = func(trial)
      File "/tmp/ipykernel_33/823259258.py", line 17, in objective
        model.fit(
      File "/opt/conda/lib/python3.10/site-packages/lightgbm/sklearn.py", line 1344, in fit
        super().fit(
      File "/opt/conda/lib/python3.10/site-packages/lightgbm/sklearn.py", line 885, in fit
        self._Booster = train(
      File "/opt/conda/lib/python3.10/site-packages/lightgbm/engine.py", line 276, in train
        booster.update(fobj=fobj)
      File "/opt/conda/lib/python3.10/site-packages/lightgbm/basic.py", line 3891, in update
        _safe_call(_LIB.LGBM_BoosterUpdateOneIter(
    KeyboardInterrupt
    [W 2024-03-08 10:51:49,588] Trial 2 failed with value None.


    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[39], line 35
         30     return np.mean(scores)
         32 study = optuna.create_study(direction='maximize',
         33                             sampler=optuna.samplers.TPESampler(seed=100) #fix random seed
         34                            )
    ---> 35 study.optimize(objective, n_trials=10)


    File /opt/conda/lib/python3.10/site-packages/optuna/study/study.py:451, in Study.optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
        348 def optimize(
        349     self,
        350     func: ObjectiveFuncType,
       (...)
        357     show_progress_bar: bool = False,
        358 ) -> None:
        359     """Optimize an objective function.
        360 
        361     Optimization is done by choosing a suitable set of hyperparameter values from a given
       (...)
        449             If nested invocation of this method occurs.
        450     """
    --> 451     _optimize(
        452         study=self,
        453         func=func,
        454         n_trials=n_trials,
        455         timeout=timeout,
        456         n_jobs=n_jobs,
        457         catch=tuple(catch) if isinstance(catch, Iterable) else (catch,),
        458         callbacks=callbacks,
        459         gc_after_trial=gc_after_trial,
        460         show_progress_bar=show_progress_bar,
        461     )


    File /opt/conda/lib/python3.10/site-packages/optuna/study/_optimize.py:66, in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
         64 try:
         65     if n_jobs == 1:
    ---> 66         _optimize_sequential(
         67             study,
         68             func,
         69             n_trials,
         70             timeout,
         71             catch,
         72             callbacks,
         73             gc_after_trial,
         74             reseed_sampler_rng=False,
         75             time_start=None,
         76             progress_bar=progress_bar,
         77         )
         78     else:
         79         if n_jobs == -1:


    File /opt/conda/lib/python3.10/site-packages/optuna/study/_optimize.py:163, in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
        160         break
        162 try:
    --> 163     frozen_trial = _run_trial(study, func, catch)
        164 finally:
        165     # The following line mitigates memory problems that can be occurred in some
        166     # environments (e.g., services that use computing containers such as GitHub Actions).
        167     # Please refer to the following PR for further details:
        168     # https://github.com/optuna/optuna/pull/325.
        169     if gc_after_trial:


    File /opt/conda/lib/python3.10/site-packages/optuna/study/_optimize.py:251, in _run_trial(study, func, catch)
        244         assert False, "Should not reach."
        246 if (
        247     frozen_trial.state == TrialState.FAIL
        248     and func_err is not None
        249     and not isinstance(func_err, catch)
        250 ):
    --> 251     raise func_err
        252 return frozen_trial


    File /opt/conda/lib/python3.10/site-packages/optuna/study/_optimize.py:200, in _run_trial(study, func, catch)
        198 with get_heartbeat_thread(trial._trial_id, study._storage):
        199     try:
    --> 200         value_or_values = func(trial)
        201     except exceptions.TrialPruned as e:
        202         # TODO(mamu): Handle multi-objective cases.
        203         state = TrialState.PRUNED


    Cell In[39], line 17, in objective(trial)
         14 #train model
         15 model = lgb.LGBMRanker(n_estimators=1000, **param, random_state=100, early_stopping_rounds=50,
         16     verbose=10)
    ---> 17 model.fit(
         18     train[features],
         19     train[target_col],
         20     group=train_query,
         21     eval_set=[(test[features], test[target_col])],
         22     eval_group=[list(test_query)],
         23     eval_at=[1, 3, 5, 10, 20], # calc validation ndcg@1,3,5,10,20
         24 )
         26 # maximize mean ndcg
         27 scores = []


    File /opt/conda/lib/python3.10/site-packages/lightgbm/sklearn.py:1344, in LGBMRanker.fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_group, eval_metric, eval_at, feature_name, categorical_feature, callbacks, init_model)
       1340         raise ValueError("Should set group for all eval datasets for ranking task; "
       1341                          "if you use dict, the index should start from 0")
       1343 self._eval_at = eval_at
    -> 1344 super().fit(
       1345     X,
       1346     y,
       1347     sample_weight=sample_weight,
       1348     init_score=init_score,
       1349     group=group,
       1350     eval_set=eval_set,
       1351     eval_names=eval_names,
       1352     eval_sample_weight=eval_sample_weight,
       1353     eval_init_score=eval_init_score,
       1354     eval_group=eval_group,
       1355     eval_metric=eval_metric,
       1356     feature_name=feature_name,
       1357     categorical_feature=categorical_feature,
       1358     callbacks=callbacks,
       1359     init_model=init_model
       1360 )
       1361 return self


    File /opt/conda/lib/python3.10/site-packages/lightgbm/sklearn.py:885, in LGBMModel.fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
        882 evals_result: _EvalResultDict = {}
        883 callbacks.append(record_evaluation(evals_result))
    --> 885 self._Booster = train(
        886     params=params,
        887     train_set=train_set,
        888     num_boost_round=self.n_estimators,
        889     valid_sets=valid_sets,
        890     valid_names=eval_names,
        891     feval=eval_metrics_callable,  # type: ignore[arg-type]
        892     init_model=init_model,
        893     feature_name=feature_name,
        894     callbacks=callbacks
        895 )
        897 self._evals_result = evals_result
        898 self._best_iteration = self._Booster.best_iteration


    File /opt/conda/lib/python3.10/site-packages/lightgbm/engine.py:276, in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, feature_name, categorical_feature, keep_training_booster, callbacks)
        268 for cb in callbacks_before_iter:
        269     cb(callback.CallbackEnv(model=booster,
        270                             params=params,
        271                             iteration=i,
        272                             begin_iteration=init_iteration,
        273                             end_iteration=init_iteration + num_boost_round,
        274                             evaluation_result_list=None))
    --> 276 booster.update(fobj=fobj)
        278 evaluation_result_list: List[_LGBM_BoosterEvalMethodResultType] = []
        279 # check evaluation result.


    File /opt/conda/lib/python3.10/site-packages/lightgbm/basic.py:3891, in Booster.update(self, train_set, fobj)
       3889 if self.__set_objective_to_none:
       3890     raise LightGBMError('Cannot update due to null objective function.')
    -> 3891 _safe_call(_LIB.LGBM_BoosterUpdateOneIter(
       3892     self._handle,
       3893     ctypes.byref(is_finished)))
       3894 self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
       3895 return is_finished.value == 1


    KeyboardInterrupt: 



```python
print('Best trial:', study.best_trial.params)
```

    Best trial: {'lambda_l1': 0.0007773998922821829, 'lambda_l2': 3.2012859298995277e-06, 'max_depth': 5, 'num_leaves': 217, 'colsample_bytree': 0.10424697057187532, 'min_child_samples': 16}



```python
# train with best params
best_params = study.best_trial.params
model = lgb.LGBMRanker(n_estimators=1000, **best_params, random_state=100,early_stopping_rounds=50,
    verbose=10)
model.fit(
    train[features],
    train[target_col],
    group=train_query,
    eval_set=[(test[features], test[target_col])],
    eval_group=[list(test_query)],
    eval_at=[1, 3, 5, 10, 20]
)
```

    [LightGBM] [Warning] lambda_l2 is set=3.2012859298995277e-06, reg_lambda=0.0 will be ignored. Current value: lambda_l2=3.2012859298995277e-06
    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] lambda_l1 is set=0.0007773998922821829, reg_alpha=0.0 will be ignored. Current value: lambda_l1=0.0007773998922821829
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Warning] lambda_l2 is set=3.2012859298995277e-06, reg_lambda=0.0 will be ignored. Current value: lambda_l2=3.2012859298995277e-06
    [LightGBM] [Warning] lambda_l1 is set=0.0007773998922821829, reg_alpha=0.0 will be ignored. Current value: lambda_l1=0.0007773998922821829
    [LightGBM] [Debug] Dataset::GetMultiBinFromSparseFeatures: sparse rate 0.883502
    [LightGBM] [Debug] Dataset::GetMultiBinFromAllFeatures: sparse rate 0.447191
    [LightGBM] [Debug] init for col-wise cost 1.088307 seconds, init for row-wise cost 3.176592 seconds
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 3.355725 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 5102
    [LightGBM] [Info] Number of data points in the train set: 4610662, number of used features: 42
    [LightGBM] [Warning] lambda_l2 is set=3.2012859298995277e-06, reg_lambda=0.0 will be ignored. Current value: lambda_l2=3.2012859298995277e-06
    [LightGBM] [Warning] early_stopping_round is set=50, early_stopping_rounds=50 will be ignored. Current value: early_stopping_round=50
    [LightGBM] [Warning] lambda_l1 is set=0.0007773998922821829, reg_alpha=0.0 will be ignored. Current value: lambda_l1=0.0007773998922821829
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    Training until validation scores don't improve for 50 rounds
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 24 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 19 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 24 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 12 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 17 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 13 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 20 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 10 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 26 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 14 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 9 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 7 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 27 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 12 and depth = 4
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 8 and depth = 3
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 23 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 28 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 25 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 16 and depth = 4
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 29 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 30 and depth = 5
    [LightGBM] [Debug] Trained a tree with leaves = 32 and depth = 5
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Debug] Trained a tree with leaves = 31 and depth = 5
    Early stopping, best iteration is:
    [181]	valid_0's ndcg@1: 0.828856	valid_0's ndcg@3: 0.891943	valid_0's ndcg@5: 0.915653	valid_0's ndcg@10: 0.932575	valid_0's ndcg@20: 0.937398





<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LGBMRanker(colsample_bytree=0.10424697057187532, early_stopping_rounds=50,
           lambda_l1=0.0007773998922821829, lambda_l2=3.2012859298995277e-06,
           max_depth=5, min_child_samples=16, n_estimators=1000, num_leaves=217,
           random_state=100, verbose=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMRanker</label><div class="sk-toggleable__content"><pre>LGBMRanker(colsample_bytree=0.10424697057187532, early_stopping_rounds=50,
           lambda_l1=0.0007773998922821829, lambda_l2=3.2012859298995277e-06,
           max_depth=5, min_child_samples=16, n_estimators=1000, num_leaves=217,
           random_state=100, verbose=10)</pre></div></div></div></div></div>



Validation NGCG@10 is 0.932575


```python
# feature imporance
plt.figure(figsize=(10, 7))
df_plt = pd.DataFrame({'feature_name': features, 'feature_importance': model.feature_importances_})
df_plt.sort_values('feature_importance', ascending=False, inplace=True)
sns.barplot(x="feature_importance", y="feature_name", data=df_plt)
plt.title('feature importance')
```




    Text(0.5, 1.0, 'feature importance')




    
![png](https://dakshjain97.github.io/assets/img/learning-to-rank-anime-recom-using-lightgbmranker_files/learning-to-rank-anime-recom-using-lightgbmranker_38_1.png)
    


# Prediction for a sample user_id & comparing with true ranking


```python
test[test['user_id']==3][features]
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
      <th>Score</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Watching</th>
      <th>Completed</th>
      <th>On-Hold</th>
      <th>Dropped</th>
      <th>Score-1</th>
      <th>Score-2</th>
      <th>Score-3</th>
      <th>Score-4</th>
      <th>Score-5</th>
      <th>Score-6</th>
      <th>Score-7</th>
      <th>Score-8</th>
      <th>Score-9</th>
      <th>Score-10</th>
      <th>rating_count</th>
      <th>rating_mean</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Comedy</th>
      <th>Slice of Life</th>
      <th>Drama</th>
      <th>Sci-Fi</th>
      <th>Game</th>
      <th>Harem</th>
      <th>Military</th>
      <th>Space</th>
      <th>Music</th>
      <th>Mecha</th>
      <th>Historical</th>
      <th>Mystery</th>
      <th>School</th>
      <th>Hentai</th>
      <th>Fantasy</th>
      <th>Horror</th>
      <th>Kids</th>
      <th>Sports</th>
      <th>Magic</th>
      <th>Romance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>7.38</td>
      <td>1954</td>
      <td>62358</td>
      <td>91</td>
      <td>2491</td>
      <td>41456</td>
      <td>1649</td>
      <td>1353</td>
      <td>75.0</td>
      <td>84.0</td>
      <td>202.0</td>
      <td>572.0</td>
      <td>1631.0</td>
      <td>4180.0</td>
      <td>9769.0</td>
      <td>8980.0</td>
      <td>3756.0</td>
      <td>2039.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
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
      <td>7.52</td>
      <td>617</td>
      <td>236200</td>
      <td>1915</td>
      <td>15056</td>
      <td>106133</td>
      <td>12171</td>
      <td>14488</td>
      <td>412.0</td>
      <td>550.0</td>
      <td>1012.0</td>
      <td>2516.0</td>
      <td>5661.0</td>
      <td>10780.0</td>
      <td>23941.0</td>
      <td>24546.0</td>
      <td>13690.0</td>
      <td>9142.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
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
      <td>8.12</td>
      <td>1095</td>
      <td>134642</td>
      <td>691</td>
      <td>3055</td>
      <td>114742</td>
      <td>1279</td>
      <td>751</td>
      <td>62.0</td>
      <td>33.0</td>
      <td>96.0</td>
      <td>293.0</td>
      <td>1244.0</td>
      <td>4454.0</td>
      <td>17122.0</td>
      <td>29970.0</td>
      <td>18903.0</td>
      <td>12292.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.85</td>
      <td>190</td>
      <td>577290</td>
      <td>8794</td>
      <td>29278</td>
      <td>437639</td>
      <td>7575</td>
      <td>9892</td>
      <td>794.0</td>
      <td>911.0</td>
      <td>2085.0</td>
      <td>4789.0</td>
      <td>11114.0</td>
      <td>28180.0</td>
      <td>78914.0</td>
      <td>114298.0</td>
      <td>69226.0</td>
      <td>43419.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>5</th>
      <td>6.31</td>
      <td>2366</td>
      <td>45815</td>
      <td>68</td>
      <td>1338</td>
      <td>25465</td>
      <td>360</td>
      <td>316</td>
      <td>253.0</td>
      <td>513.0</td>
      <td>913.0</td>
      <td>1412.0</td>
      <td>2547.0</td>
      <td>4085.0</td>
      <td>5136.0</td>
      <td>2836.0</td>
      <td>1240.0</td>
      <td>755.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.29</td>
      <td>21</td>
      <td>1478842</td>
      <td>52618</td>
      <td>106570</td>
      <td>1130438</td>
      <td>30627</td>
      <td>34941</td>
      <td>3590.0</td>
      <td>2835.0</td>
      <td>5128.0</td>
      <td>11552.0</td>
      <td>21241.0</td>
      <td>47623.0</td>
      <td>133043.0</td>
      <td>267047.0</td>
      <td>267437.0</td>
      <td>192801.0</td>
      <td>45</td>
      <td>7.777778</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.10</td>
      <td>604</td>
      <td>239361</td>
      <td>412</td>
      <td>3430</td>
      <td>207405</td>
      <td>1022</td>
      <td>667</td>
      <td>242.0</td>
      <td>215.0</td>
      <td>384.0</td>
      <td>882.0</td>
      <td>2952.0</td>
      <td>8042.0</td>
      <td>27871.0</td>
      <td>52711.0</td>
      <td>36765.0</td>
      <td>19307.0</td>
      <td>45</td>
      <td>7.777778</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.56</td>
      <td>277</td>
      <td>463838</td>
      <td>3720</td>
      <td>17602</td>
      <td>332808</td>
      <td>8950</td>
      <td>10341</td>
      <td>641.0</td>
      <td>936.0</td>
      <td>2069.0</td>
      <td>4989.0</td>
      <td>13154.0</td>
      <td>28899.0</td>
      <td>65743.0</td>
      <td>70903.0</td>
      <td>39804.0</td>
      <td>25000.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.17</td>
      <td>197</td>
      <td>566538</td>
      <td>9555</td>
      <td>8258</td>
      <td>434086</td>
      <td>3491</td>
      <td>2174</td>
      <td>1067.0</td>
      <td>992.0</td>
      <td>1737.0</td>
      <td>4322.0</td>
      <td>9462.0</td>
      <td>22369.0</td>
      <td>54906.0</td>
      <td>91288.0</td>
      <td>82148.0</td>
      <td>77192.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <th>10</th>
      <td>6.78</td>
      <td>1016</td>
      <td>146642</td>
      <td>328</td>
      <td>6073</td>
      <td>103539</td>
      <td>2981</td>
      <td>4262</td>
      <td>449.0</td>
      <td>706.0</td>
      <td>1478.0</td>
      <td>3349.0</td>
      <td>9050.0</td>
      <td>15661.0</td>
      <td>22859.0</td>
      <td>13666.0</td>
      <td>6350.0</td>
      <td>4096.0</td>
      <td>45</td>
      <td>7.777778</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Comparing predicted score vs actual rating
pd.concat([pd.DataFrame(model.predict(test[test['user_id']==3][features]),columns = ['predicted_score']),
test[test['user_id']==3][['rating']].reset_index(drop = True)],axis = 1).sort_values(by = ['predicted_score'],ascending = False)
```

    [LightGBM] [Warning] lambda_l2 is set=3.2012859298995277e-06, reg_lambda=0.0 will be ignored. Current value: lambda_l2=3.2012859298995277e-06
    [LightGBM] [Warning] lambda_l1 is set=0.0007773998922821829, reg_alpha=0.0 will be ignored. Current value: lambda_l1=0.0007773998922821829





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
      <th>predicted_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.654916</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.407829</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.121365</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.079490</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.394757</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.807257</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.820733</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1.203161</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-2.222275</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.477007</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



In above example we observe anime titles having highes predicted score also has higher true ratings , meaning titles which are ranked higher in postiion have higher true ranks (validation NDCG@10>0.93)


```python

```
