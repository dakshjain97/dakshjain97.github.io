---
layout: post
title: Recommendation using collaborative filtering by item & user similarity
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [recommendation, collaborative filtering, user & item similarities, embedding, deep learning, dot product]
author: Daksh Jain
---
In this notebook objective is to demonstrate collaborative filtering based recommendation techniques to recommend anime by utilizing user & item based similarities . To achieve this a Neural network model is created with two embedding layers (each for user & item) , for POC purose model is only trained on small epoch. To calculate similarities dot product (cosine similarity) is used among user & item embedding vectors


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
#importing libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, Dot, Reshape, Dense, Flatten
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
```

    2024-03-13 12:27:31.431391: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-03-13 12:27:31.431535: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-03-13 12:27:31.602701: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered


# Data Cleaning & Loading


```python
df_user = pd.read_csv('/kaggle/input/anime-recommendation-database-2020/animelist.csv',usecols=["user_id", "anime_id", "rating"])
```


```python
df_user.head()
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
      <td>67</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6702</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>242</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4898</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>21</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_user.shape
```




    (109224747, 3)




```python
#consider only users rated >400 animes (to limit data size) 
df_cnt_ratings = df_user.groupby(['user_id'])['rating'].count().to_frame()
df_user = df_user[df_user['user_id'].isin(df_cnt_ratings[df_cnt_ratings['rating']>=400].index)].reset_index(drop = True)
```


```python
df_user.shape
```




    (71418114, 3)




```python
#min max scaling rating column
max_value = df_user['rating'].max()
min_value = df_user['rating'].min()
df_user['rating'] = df_user['rating'].apply(lambda x: (x-min_value)/(max_value-min_value)).astype(float)
```


```python
df_user[df_user['rating']>0].head()
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
      <th>1</th>
      <td>2</td>
      <td>235</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>5042</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>7593</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>21</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>22</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# Data preprocessing 

Creating user & anime ids mappings to index values


```python
user_ids = df_user["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
df_user["user"] = df_user["user_id"].map(user2user_encoded)
```


```python
anime_ids = df_user["anime_id"].unique().tolist()
anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}
df_user["anime"] = df_user["anime_id"].map(anime2anime_encoded)

```


```python
df_user.head()
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
      <th>user</th>
      <th>anime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>24833</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>235</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>36721</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>40956</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>31933</td>
      <td>0.0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Shuffle
df_user = df_user.sample(frac=1, random_state=73)

X = df_user[['user', 'anime']].values
y = df_user["rating"]
```


```python
X
```




    array([[59285,  1557],
           [12706,  3809],
           [54050,  3706],
           ...,
           [65683,  1988],
           [77931,  1746],
           [40652,  7807]])




```python
y
```




    46193982    0.7
    9912332     0.7
    42088699    0.0
    64470340    1.0
    58655135    0.7
               ... 
    41208174    0.8
    70897987    0.8
    51101456    0.0
    60760458    0.0
    31657902    0.0
    Name: rating, Length: 71418114, dtype: float64




```python
# Split
test_set_size = 10000 #10k for test set
train_indices = df_user.shape[0] - test_set_size 

X_train, X_test, y_train, y_test = (
    X[:train_indices],
    X[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

```


```python
X_train.shape
```




    (71408114, 2)




```python
X_train
```




    array([[59285,  1557],
           [12706,  3809],
           [54050,  3706],
           ...,
           [34207,  3417],
           [58411,  3233],
           [58969,  3412]])




```python
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]
```


```python

```

# Embedding/Model Creation


```python
n_users = len(user2user_encoded)
n_animes = len(anime2anime_encoded)

embedding_size = 128
user = Input(name = 'user', shape = [1])
user_embedding = Embedding(name = 'user_embedding',
                       input_dim = n_users, #vocabolary size
                       output_dim = embedding_size)(user)

anime = Input(name = 'anime', shape = [1])
anime_embedding = Embedding(name = 'anime_embedding',
                   input_dim = n_animes, 
                   output_dim = embedding_size)(anime)

x = Dot(name = 'dot_product', normalize = True, axes = 2)([user_embedding, anime_embedding])
x = Flatten()(x)
x = Dense(1, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation("sigmoid")(x)
model = Model(inputs=[user, anime], outputs=x)
model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='Adam')
```


```python
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ user (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ anime (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ user_embedding      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │ <span style="color: #00af00; text-decoration-color: #00af00">11,730,048</span> │ user[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]        │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)         │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ anime_embedding     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │  <span style="color: #00af00; text-decoration-color: #00af00">2,247,680</span> │ anime[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)         │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dot_product (<span style="color: #0087ff; text-decoration-color: #0087ff">Dot</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ user_embedding[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│                     │                   │            │ anime_embedding[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dot_product[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ flatten[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalization │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">4</span> │ dense[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,977,734</span> (53.32 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,977,732</span> (53.32 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2</span> (8.00 B)
</pre>




```python
start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
batch_size = 10000

rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr


lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)

checkpoint_filepath = '.weights.h5'

model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True,
                                        monitor='val_loss',
                                        mode='min',
                                        save_best_only=True)

early_stopping = EarlyStopping(patience = 3, monitor='val_loss', 
                               mode='min', restore_best_weights=True)

my_callbacks = [
    model_checkpoints,
    lr_callback,
    early_stopping,   
]

```


```python
del df_user
```


```python
# Model training
history = model.fit(
    x=X_train_array,
    y=y_train,
    batch_size=batch_size,
    epochs=1,
    verbose=1,
    validation_data=(X_test_array, y_test),
    callbacks=my_callbacks
)

model.load_weights(checkpoint_filepath)
```

    [1m7141/7141[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1152s[0m 160ms/step - loss: 0.7916 - mae: 0.3827 - mse: 0.1961 - val_loss: 0.7749 - val_mae: 0.3807 - val_mse: 0.1910 - learning_rate: 1.0000e-05


Note here we are only training for 1 epoch for POC purpose , recommendations can be improved it trained on higher no of epochs

Extracting embedding weights & normalizing it


```python
def extract_weights(name, model):
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights

anime_weights = extract_weights('anime_embedding', model)
user_weights = extract_weights('user_embedding', model)
```


```python
anime_weights.shape
```




    (17560, 128)




```python
user_weights.shape
```




    (91641, 128)




```python

```

# Anime metadata


```python
df = pd.read_csv('/kaggle/input/anime-recommendation-database-2020/anime.csv', low_memory=True)
df = df.replace("Unknown", np.nan)
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
      <th>...</th>
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
      <td>...</td>
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
      <td>NaN</td>
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
<p>5 rows × 35 columns</p>
</div>




```python
df.shape
```




    (17562, 35)




```python
# Fixing Names
def getAnimeName(anime_id):
    try:
        name = df[df.anime_id == anime_id].eng_version.values[0]
        if name is np.nan:
            name = df[df.anime_id == anime_id].Name.values[0]
    except:
        print('error')
    
    return name

df['anime_id'] = df['MAL_ID']
df["eng_version"] = df['English name']
df['eng_version'] = df.anime_id.apply(lambda x: getAnimeName(x))

df.sort_values(by=['Score'], 
               inplace=True,
               ascending=False, 
               kind='quicksort',
               na_position='last')

df = df[["anime_id", "eng_version", 
         "Score", "Genres", "Episodes", 
         "Type", "Premiered", "Members"]]
```


```python
def getAnimeFrame(anime):
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        return df[df.eng_version == anime]
```


```python
cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
sypnopsis_df = pd.read_csv('/kaggle/input/anime-recommendation-database-2020/anime_with_synopsis.csv', usecols=cols)
```


```python
sypnopsis_df
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
      <th>Genres</th>
      <th>sypnopsis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>In the year 2071, humanity has colonized sever...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>other day, another bounty—such is the life of ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Trigun</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>Vash the Stampede is the man with a $$60,000,0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Witch Hunter Robin</td>
      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>
      <td>ches are individuals with special powers like ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Bouken Ou Beet</td>
      <td>Adventure, Fantasy, Shounen, Supernatural</td>
      <td>It is the dark century and the people are suff...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16209</th>
      <td>48481</td>
      <td>Daomu Biji Zhi Qinling Shen Shu</td>
      <td>Adventure, Mystery, Supernatural</td>
      <td>No synopsis information has been added to this...</td>
    </tr>
    <tr>
      <th>16210</th>
      <td>48483</td>
      <td>Mieruko-chan</td>
      <td>Comedy, Horror, Supernatural</td>
      <td>ko is a typical high school student whose life...</td>
    </tr>
    <tr>
      <th>16211</th>
      <td>48488</td>
      <td>Higurashi no Naku Koro ni Sotsu</td>
      <td>Mystery, Dementia, Horror, Psychological, Supe...</td>
      <td>Sequel to Higurashi no Naku Koro ni Gou .</td>
    </tr>
    <tr>
      <th>16212</th>
      <td>48491</td>
      <td>Yama no Susume: Next Summit</td>
      <td>Adventure, Slice of Life, Comedy</td>
      <td>New Yama no Susume anime.</td>
    </tr>
    <tr>
      <th>16213</th>
      <td>48492</td>
      <td>Scarlet Nexus</td>
      <td>Action, Fantasy</td>
      <td>Solar calendar year 2020: grotesque organisms ...</td>
    </tr>
  </tbody>
</table>
<p>16214 rows × 4 columns</p>
</div>




```python
def getSypnopsis(anime):
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]
```

# Finding Similar Animes (Item Based Recommendation)

Finding similar movies by doing dot product (cosine similarity) between movie embedding matrix


```python
def find_similar_animes(name, n=10, return_dist=False, neg=False):
    try:
        index = getAnimeFrame(name).anime_id.values[0]
        encoded_index = anime2anime_encoded.get(index)
        weights = anime_weights
        
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        n = n + 1            
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        print('animes closest to {}'.format(name))

        if return_dist:
            return dists, closest
        
        rindex = df

        SimilarityArr = []

        for close in closest:
            decoded_id = anime_encoded2anime.get(close)
            sypnopsis = getSypnopsis(decoded_id)
            anime_frame = getAnimeFrame(decoded_id)
            
            anime_name = anime_frame.eng_version.values[0]
            genre = anime_frame.Genres.values[0]
            similarity = dists[close]
            SimilarityArr.append({"anime_id": decoded_id, "name": anime_name,
                                  "similarity": similarity,"genre": genre,
                                  'sypnopsis': sypnopsis})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
        return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)

    except:
        print('{}!, Not Found in Anime list'.format(name))
```


```python
find_similar_animes('Dragon Ball Z', n=5, neg=False)
```

    animes closest to Dragon Ball Z





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
      <th>name</th>
      <th>similarity</th>
      <th>genre</th>
      <th>sypnopsis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Strike Witches: Road to Berlin</td>
      <td>0.357820</td>
      <td>Action, Military, Sci-Fi, Magic, Ecchi</td>
      <td>No synopsis information has been added to this...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pokemon XY: New Year's Eve 2014 Super Mega Spe...</td>
      <td>0.331550</td>
      <td>Action, Adventure, Comedy, Fantasy, Kids</td>
      <td>cap episode.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pokemon the Movie:Volcanion and the Mechanical...</td>
      <td>0.329195</td>
      <td>Adventure, Kids, Fantasy</td>
      <td>mysterious force binds Ash to the Mythical Pok...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kobo-chan no Koutsuu Anzen</td>
      <td>0.320661</td>
      <td>Kids</td>
      <td>affic safety film starring the cast of Kobo-ch...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Lupin III:Jigen's Gravestone</td>
      <td>0.315366</td>
      <td>Action, Adventure, Drama, Seinen</td>
      <td>The film will be a continuation spinoff of the...</td>
    </tr>
  </tbody>
</table>
</div>



# Finding similar users (user based recommendation)

Finding similar users by doing dot product (cosine similarity) between user embedding matrix


```python
def find_similar_users(item_input, n=10,return_dist=False, neg=False):
    try:
        index = item_input
        encoded_index = user2user_encoded.get(index)
        weights = user_weights
    
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        n = n + 1
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        print('> users similar to #{}'.format(item_input))

        if return_dist:
            return dists, closest
        
        rindex = df
        SimilarityArr = []
        
        for close in closest:
            similarity = dists[close]

            if isinstance(item_input, int):
                decoded_id = user_encoded2user.get(close)
                SimilarityArr.append({"similar_users": decoded_id, 
                                      "similarity": similarity})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", 
                                                        ascending=False)
        
        return Frame
    
    except:
        print('{}!, Not Found in User list'.format(name))
```


```python
find_similar_users(int('352464'), n=5, neg=False)
```

    > users similar to #352464





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
      <th>similar_users</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>352464</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>212854</td>
      <td>0.375813</td>
    </tr>
    <tr>
      <th>3</th>
      <td>226926</td>
      <td>0.373433</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135181</td>
      <td>0.367671</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42307</td>
      <td>0.355450</td>
    </tr>
    <tr>
      <th>0</th>
      <td>6148</td>
      <td>0.351089</td>
    </tr>
  </tbody>
</table>
</div>



For similar users we can get top movies rated above pct75 of rating for that user & recommend that to target user


```python

```
