---
layout: post
title: Movie Recommendation using popularity & content based techniques
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [recommendation, TFIDF, Cosine Similarity]
author: Daksh Jain
---
In this notebook objective is to demonstrate popularity based and content based recommendation techniques to recommend movie . For content based recommendations TF-IDF + cosine similarity is used to find top 10 similar movies for a movie id by using plot , actors , directors , plot keywords & genre data



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

    /kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv
    /kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv



```python
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
```


```python
df_movie = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
df_credits = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')
```


```python
df_movie.head()
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
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>[{"iso_3166_1": "GB", "name": "United Kingdom"...</td>
      <td>2015-10-26</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
    </tr>
    <tr>
      <th>3</th>
      <td>250000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 80, "nam...</td>
      <td>http://www.thedarkknightrises.com/</td>
      <td>49026</td>
      <td>[{"id": 849, "name": "dc comics"}, {"id": 853,...</td>
      <td>en</td>
      <td>The Dark Knight Rises</td>
      <td>Following the death of District Attorney Harve...</td>
      <td>112.312950</td>
      <td>[{"name": "Legendary Pictures", "id": 923}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2012-07-16</td>
      <td>1084939099</td>
      <td>165.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>The Legend Ends</td>
      <td>The Dark Knight Rises</td>
      <td>7.6</td>
      <td>9106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>260000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://movies.disney.com/john-carter</td>
      <td>49529</td>
      <td>[{"id": 818, "name": "based on novel"}, {"id":...</td>
      <td>en</td>
      <td>John Carter</td>
      <td>John Carter is a war-weary, former military ca...</td>
      <td>43.926995</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}]</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2012-03-07</td>
      <td>284139100</td>
      <td>132.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Lost in our world, found in another.</td>
      <td>John Carter</td>
      <td>6.1</td>
      <td>2124</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_credits.head()
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
      <th>movie_id</th>
      <th>title</th>
      <th>cast</th>
      <th>crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>Avatar</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>206647</td>
      <td>Spectre</td>
      <td>[{"cast_id": 1, "character": "James Bond", "cr...</td>
      <td>[{"credit_id": "54805967c3a36829b5002c41", "de...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49026</td>
      <td>The Dark Knight Rises</td>
      <td>[{"cast_id": 2, "character": "Bruce Wayne / Ba...</td>
      <td>[{"credit_id": "52fe4781c3a36847f81398c3", "de...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49529</td>
      <td>John Carter</td>
      <td>[{"cast_id": 5, "character": "John Carter", "c...</td>
      <td>[{"credit_id": "52fe479ac3a36847f813eaa3", "de...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_movie.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4803 entries, 0 to 4802
    Data columns (total 20 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   budget                4803 non-null   int64  
     1   genres                4803 non-null   object 
     2   homepage              1712 non-null   object 
     3   id                    4803 non-null   int64  
     4   keywords              4803 non-null   object 
     5   original_language     4803 non-null   object 
     6   original_title        4803 non-null   object 
     7   overview              4800 non-null   object 
     8   popularity            4803 non-null   float64
     9   production_companies  4803 non-null   object 
     10  production_countries  4803 non-null   object 
     11  release_date          4802 non-null   object 
     12  revenue               4803 non-null   int64  
     13  runtime               4801 non-null   float64
     14  spoken_languages      4803 non-null   object 
     15  status                4803 non-null   object 
     16  tagline               3959 non-null   object 
     17  title                 4803 non-null   object 
     18  vote_average          4803 non-null   float64
     19  vote_count            4803 non-null   int64  
    dtypes: float64(3), int64(4), object(13)
    memory usage: 750.6+ KB



```python
df_credits.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4803 entries, 0 to 4802
    Data columns (total 4 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   movie_id  4803 non-null   int64 
     1   title     4803 non-null   object
     2   cast      4803 non-null   object
     3   crew      4803 non-null   object
    dtypes: int64(1), object(3)
    memory usage: 150.2+ KB



```python

```

# Recommendation using popularity based method

In this method recommendation are same across all users & items (movies in this case) , i.e Most watched / popular movies


```python
df_movie[['popularity','original_title']].sort_values(by = ['popularity'],ascending = False).head(10).set_index('original_title').plot(kind = 'bar', )
```




    <Axes: xlabel='original_title'>




    
![png](https://dakshjain97.github.io/assets/img/recommendation-popularity-content-based_files/recommendation-popularity-content-based_10_1.png)
    



```python

```

# Recommendation using content based method

In this case each movie can have different set of recommendations on basis of content (e.g cast, plot etc)


```python
df_merged = pd.merge(df_movie,df_credits,left_on = ['id'],right_on = ['movie_id'],how = 'inner')
```

Recommendation using plot of each movie by creating TF-IDF matrix for each movies plot & using cosine similarity


```python
df_merged[['id','original_title','overview']].head()
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
      <th>id</th>
      <th>original_title</th>
      <th>overview</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>206647</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49026</td>
      <td>The Dark Knight Rises</td>
      <td>Following the death of District Attorney Harve...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49529</td>
      <td>John Carter</td>
      <td>John Carter is a war-weary, former military ca...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df_merged['overview'] = df_merged['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df_merged['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape
```




    (4803, 20978)




```python
tfidf_matrix.todense()
```




    matrix([[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]])




```python
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```


```python
cosine_sim.shape
```




    (4803, 4803)




```python
cosine_sim
```




    array([[1.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 1.        , 0.        , ..., 0.02160533, 0.        ,
            0.        ],
           [0.        , 0.        , 1.        , ..., 0.01488159, 0.        ,
            0.        ],
           ...,
           [0.        , 0.02160533, 0.01488159, ..., 1.        , 0.01609091,
            0.00701914],
           [0.        , 0.        , 0.        , ..., 0.01609091, 1.        ,
            0.01171696],
           [0.        , 0.        , 0.        , ..., 0.00701914, 0.01171696,
            1.        ]])




```python
#title to index mapping
maping = df_merged[['original_title','id']].reset_index().set_index('original_title')
```


```python
maping
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
      <th>index</th>
      <th>id</th>
    </tr>
    <tr>
      <th>original_title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avatar</th>
      <td>0</td>
      <td>19995</td>
    </tr>
    <tr>
      <th>Pirates of the Caribbean: At World's End</th>
      <td>1</td>
      <td>285</td>
    </tr>
    <tr>
      <th>Spectre</th>
      <td>2</td>
      <td>206647</td>
    </tr>
    <tr>
      <th>The Dark Knight Rises</th>
      <td>3</td>
      <td>49026</td>
    </tr>
    <tr>
      <th>John Carter</th>
      <td>4</td>
      <td>49529</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>El Mariachi</th>
      <td>4798</td>
      <td>9367</td>
    </tr>
    <tr>
      <th>Newlyweds</th>
      <td>4799</td>
      <td>72766</td>
    </tr>
    <tr>
      <th>Signed, Sealed, Delivered</th>
      <td>4800</td>
      <td>231617</td>
    </tr>
    <tr>
      <th>Shanghai Calling</th>
      <td>4801</td>
      <td>126186</td>
    </tr>
    <tr>
      <th>My Date with Drew</th>
      <td>4802</td>
      <td>25975</td>
    </tr>
  </tbody>
</table>
<p>4803 rows × 2 columns</p>
</div>




```python
df_merged.loc[0,'original_title']
```




    'Avatar'




```python
def content_based_recomm(title, n = 10):
    sim = cosine_sim[maping.loc[title,'index']]
    sim_sco = list(enumerate(sim))
    top_10_sim = sorted(sim_sco, key=lambda x: x[1], reverse=True)[1:11]
    return [df_merged.loc[x[0],'original_title'] for x in top_10_sim]
```


```python
content_based_recomm("The Dark Knight Rises")
```




    ['The Dark Knight',
     'Batman Forever',
     'Batman Returns',
     'Batman',
     'Batman: The Dark Knight Returns, Part 2',
     'Batman Begins',
     'Slow Burn',
     'Batman v Superman: Dawn of Justice',
     'JFK',
     'Batman & Robin']




```python

```

Here we notice top 10 similar movies on basis of movie plot are all series of batman , we can improve this by providing more diverse recomm using actors, directors , genres & plot keywords data


```python
feats = ['cast', 'crew', 'keywords', 'genres']
```


```python
df_merged[['original_title']+feats].head()
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
      <th>original_title</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avatar</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spectre</td>
      <td>[{"cast_id": 1, "character": "James Bond", "cr...</td>
      <td>[{"credit_id": "54805967c3a36829b5002c41", "de...</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight Rises</td>
      <td>[{"cast_id": 2, "character": "Bruce Wayne / Ba...</td>
      <td>[{"credit_id": "52fe4781c3a36847f81398c3", "de...</td>
      <td>[{"id": 849, "name": "dc comics"}, {"id": 853,...</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 80, "nam...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>John Carter</td>
      <td>[{"cast_id": 5, "character": "John Carter", "c...</td>
      <td>[{"credit_id": "52fe479ac3a36847f813eaa3", "de...</td>
      <td>[{"id": 818, "name": "based on novel"}, {"id":...</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#converting columns to usable data strutures
for col in feats:
    df_merged[col] = df_merged[col].apply(literal_eval)
```


```python
df_merged[['original_title']+feats].head()
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
      <th>original_title</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avatar</td>
      <td>[{'cast_id': 242, 'character': 'Jake Sully', '...</td>
      <td>[{'credit_id': '52fe48009251416c750aca23', 'de...</td>
      <td>[{'id': 1463, 'name': 'culture clash'}, {'id':...</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 12, 'nam...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[{'cast_id': 4, 'character': 'Captain Jack Spa...</td>
      <td>[{'credit_id': '52fe4232c3a36847f800b579', 'de...</td>
      <td>[{'id': 270, 'name': 'ocean'}, {'id': 726, 'na...</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spectre</td>
      <td>[{'cast_id': 1, 'character': 'James Bond', 'cr...</td>
      <td>[{'credit_id': '54805967c3a36829b5002c41', 'de...</td>
      <td>[{'id': 470, 'name': 'spy'}, {'id': 818, 'name...</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 12, 'nam...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight Rises</td>
      <td>[{'cast_id': 2, 'character': 'Bruce Wayne / Ba...</td>
      <td>[{'credit_id': '52fe4781c3a36847f81398c3', 'de...</td>
      <td>[{'id': 849, 'name': 'dc comics'}, {'id': 853,...</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 80, 'nam...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>John Carter</td>
      <td>[{'cast_id': 5, 'character': 'John Carter', 'c...</td>
      <td>[{'credit_id': '52fe479ac3a36847f813eaa3', 'de...</td>
      <td>[{'id': 818, 'name': 'based on novel'}, {'id':...</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 12, 'nam...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merged['director'] = df_merged['crew'].apply(lambda x: [l['name'] for l in x if l['job']=='Director'])
```


```python
# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
```


```python
for col in ['cast', 'keywords', 'genres']:
    df_merged[col] = df_merged[col].apply(get_list)
```


```python
feats = feats = ['cast', 'director', 'keywords', 'genres']
df_merged[['original_title']+feats].head()
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
      <th>original_title</th>
      <th>cast</th>
      <th>director</th>
      <th>keywords</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avatar</td>
      <td>[Sam Worthington, Zoe Saldana, Sigourney Weaver]</td>
      <td>[James Cameron]</td>
      <td>[culture clash, future, space war]</td>
      <td>[Action, Adventure, Fantasy]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[Johnny Depp, Orlando Bloom, Keira Knightley]</td>
      <td>[Gore Verbinski]</td>
      <td>[ocean, drug abuse, exotic island]</td>
      <td>[Adventure, Fantasy, Action]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spectre</td>
      <td>[Daniel Craig, Christoph Waltz, Léa Seydoux]</td>
      <td>[Sam Mendes]</td>
      <td>[spy, based on novel, secret agent]</td>
      <td>[Action, Adventure, Crime]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight Rises</td>
      <td>[Christian Bale, Michael Caine, Gary Oldman]</td>
      <td>[Christopher Nolan]</td>
      <td>[dc comics, crime fighter, terrorist]</td>
      <td>[Action, Crime, Drama]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>John Carter</td>
      <td>[Taylor Kitsch, Lynn Collins, Samantha Morton]</td>
      <td>[Andrew Stanton]</td>
      <td>[based on novel, mars, medallion]</td>
      <td>[Action, Adventure, Science Fiction]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        return ''
```


```python
for col in ['cast', 'keywords', 'genres','director']:
    df_merged[col] = df_merged[col].apply(clean_data)
```


```python
df_merged[['original_title']+feats].head()
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
      <th>original_title</th>
      <th>cast</th>
      <th>director</th>
      <th>keywords</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avatar</td>
      <td>[samworthington, zoesaldana, sigourneyweaver]</td>
      <td>[jamescameron]</td>
      <td>[cultureclash, future, spacewar]</td>
      <td>[action, adventure, fantasy]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[johnnydepp, orlandobloom, keiraknightley]</td>
      <td>[goreverbinski]</td>
      <td>[ocean, drugabuse, exoticisland]</td>
      <td>[adventure, fantasy, action]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spectre</td>
      <td>[danielcraig, christophwaltz, léaseydoux]</td>
      <td>[sammendes]</td>
      <td>[spy, basedonnovel, secretagent]</td>
      <td>[action, adventure, crime]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight Rises</td>
      <td>[christianbale, michaelcaine, garyoldman]</td>
      <td>[christophernolan]</td>
      <td>[dccomics, crimefighter, terrorist]</td>
      <td>[action, crime, drama]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>John Carter</td>
      <td>[taylorkitsch, lynncollins, samanthamorton]</td>
      <td>[andrewstanton]</td>
      <td>[basedonnovel, mars, medallion]</td>
      <td>[action, adventure, sciencefiction]</td>
    </tr>
  </tbody>
</table>
</div>




```python
#combining all values into a string
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])

df_merged['soup'] = df_merged.apply(create_soup, axis=1)
```


```python
df_merged['soup'][0]
```




    'cultureclash future spacewar samworthington zoesaldana sigourneyweaver jamescameron action adventure fantasy'




```python
#using tf-idf to find most similar movies

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df_merged['soup'] = df_merged['soup'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df_merged['soup'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape
```




    (4803, 11742)




```python
tfidf_matrix.todense()
```




    matrix([[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]])




```python
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```


```python
cosine_sim
```




    array([[1.        , 0.07121855, 0.03721726, ..., 0.        , 0.        ,
            0.        ],
           [0.07121855, 1.        , 0.03911498, ..., 0.        , 0.        ,
            0.        ],
           [0.03721726, 0.03911498, 1.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 1.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 1.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            1.        ]])




```python
content_based_recomm("The Dark Knight Rises")
```




    ['The Dark Knight',
     'Batman Begins',
     'The Prestige',
     "Amidst the Devil's Wings",
     'The Killer Inside Me',
     'Insomnia',
     'Interstellar',
     'The Statement',
     'Inception',
     'London Has Fallen']



only using plot keywords result

['The Dark Knight',
 'Batman Forever',
 'Batman Returns',
 'Batman',
 'Batman: The Dark Knight Returns, Part 2',
 'Batman Begins',
 'Slow Burn',
 'Batman v Superman: Dawn of Justice',
 'JFK',
 'Batman & Robin']


```python

```


```python

```
