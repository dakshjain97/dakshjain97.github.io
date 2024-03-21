---
layout: post
title: Image Classification using transfer learning RESNET-50
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [image classification, RESNET 50, keras, transfer learning]
author: Daksh Jain
---
This notebook demonstrates transfer learning using resenet-50 model using keras framework for a image classification task . The task is to classify images into rural / urban areas 

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

    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural30.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural25.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural40.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural20.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural0.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural35.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural10.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural15.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural5.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/rural/rural45.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_5.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_25.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_45.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_20.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban0.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_15.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_30.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_10.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_40.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/val/urban/urban_35.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural31.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural44.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural38.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural27.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural32.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural24.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural28.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural43.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural11.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural1.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural7.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural33.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural23.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural9.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural37.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural26.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural21.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural8.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural22.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural6.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural29.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural4.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural34.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural41.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural3.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural16.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural2.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural39.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural18.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural12.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural14.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural17.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural19.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural13.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural42.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/rural/rural36.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_22.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_18.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_2.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_6.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_1.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_29.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_11.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_34.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_42.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_4.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_41.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_39.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_16.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_38.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_26.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_7.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_28.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_44.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_9.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_43.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_31.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_21.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_37.jpg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_24.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_32.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_36.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_27.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_3.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_19.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_12.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_13.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_17.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_8.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_23.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_33.jpeg
    /kaggle/input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_14.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural30.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural25.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural40.jpg
    /kaggle/input/urban-and-rural-photos/val/rural/rural20.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural0.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural35.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural10.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural15.jpg
    /kaggle/input/urban-and-rural-photos/val/rural/rural5.jpeg
    /kaggle/input/urban-and-rural-photos/val/rural/rural45.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_5.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_25.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_45.jpg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_20.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban0.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_15.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_30.jpg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_10.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_40.jpeg
    /kaggle/input/urban-and-rural-photos/val/urban/urban_35.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural31.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural44.jpg
    /kaggle/input/urban-and-rural-photos/train/rural/rural38.jpg
    /kaggle/input/urban-and-rural-photos/train/rural/rural27.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural32.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural24.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural28.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural43.jpg
    /kaggle/input/urban-and-rural-photos/train/rural/rural11.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural1.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural7.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural33.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural23.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural9.jpg
    /kaggle/input/urban-and-rural-photos/train/rural/rural37.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural26.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural21.jpg
    /kaggle/input/urban-and-rural-photos/train/rural/rural8.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural22.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural6.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural29.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural4.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural34.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural41.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural3.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural16.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural2.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural39.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural18.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural12.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural14.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural17.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural19.jpg
    /kaggle/input/urban-and-rural-photos/train/rural/rural13.jpeg
    /kaggle/input/urban-and-rural-photos/train/rural/rural42.jpg
    /kaggle/input/urban-and-rural-photos/train/rural/rural36.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_22.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_18.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_2.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_6.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_1.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_29.jpg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_11.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_34.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_42.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_4.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_41.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_39.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_16.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_38.jpg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_26.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_7.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_28.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_44.jpg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_9.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_43.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_31.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_21.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_37.jpg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_24.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_32.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_36.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_27.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_3.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_19.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_12.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_13.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_17.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_8.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_23.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_33.jpeg
    /kaggle/input/urban-and-rural-photos/train/urban/urban_14.jpeg



```python
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

# Data Loading


```python
ds = imread('/kaggle/input/urban-and-rural-photos/val/urban/urban_25.jpeg')
plt.imshow(ds)
```




    <matplotlib.image.AxesImage at 0x781d95a53a60>




    
![png](https://dakshjain97.github.io/assets/img/transfer-learning-image-classification-resnet-50_files/transfer-learning-image-classification-resnet-50_3_1.png)
    



```python
ds.shape
```




    (225, 225, 3)




```python
ds = imread('/kaggle/input/urban-and-rural-photos/val/rural/rural45.jpeg')
plt.imshow(ds)
```




    <matplotlib.image.AxesImage at 0x781d9588c2b0>




    
![png](https://dakshjain97.github.io/assets/img/transfer-learning-image-classification-resnet-50_files/transfer-learning-image-classification-resnet-50_5_1.png)
    



```python
ds.shape
```




    (225, 225, 3)



The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.


```python
image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/val',
        target_size=(image_size, image_size),
        batch_size=20,
        class_mode='categorical')
```

    Found 72 images belonging to 2 classes.
    Found 20 images belonging to 2 classes.


# Model Building


```python
num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ resnet50 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)           │ ?                      │    <span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> (89.98 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> (89.98 MB)
</pre>




```python
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
train_generator.labels
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1], dtype=int32)




```python
my_new_model.fit(
        train_generator,
        steps_per_epoch=6, # 72 train images / 12 batch size
        validation_data=validation_generator,
        validation_steps=1) # 20 validation images / 20 batch size
```

    /opt/conda/lib/python3.10/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:122: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
      self._warn_if_super_not_called()


    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 2s/step - accuracy: 0.7911 - loss: 0.5092 - val_accuracy: 0.9500 - val_loss: 0.1249





    <keras.src.callbacks.history.History at 0x781d289d0910>



Here we get 95% validation accuracy 


```python

```
