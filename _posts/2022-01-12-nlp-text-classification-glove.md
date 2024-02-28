---
layout: post
title: NLP Text Classification using Glove, Tf-IDF & LSTM
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [NLP, text classification, word embeddings, tf-idf, deep learning, LSTM]
author: Daksh Jain
---
This notebooks objective is to showcase & compare text classification results using pre-trained word embedings, TF-IDF & deep learning model like LSTM . Here objective is to classify text into 3 categories (authors). Using TF-IDF gave highest f1-score , but LSTM also was close that too without using pre-trained word embeddings.

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

    /kaggle/input/spooky/sample_submission.csv
    /kaggle/input/spooky/train.csv
    /kaggle/input/spooky/test.csv
    /kaggle/input/glove840b300dtxt/glove.840B.300d.txt



```python
#Importing libraries
import matplotlib.pyplot as plt
import re
from collections import Counter
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
import subprocess
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm.pandas()

#downloading nltk
nltk.download('wordnet', download_dir='/kaggle/working/')
command = "unzip /kaggle/working/corpora/wordnet.zip -d /kaggle/working/corpora"
subprocess.run(command.split())
nltk.data.path.append('/kaggle/working/')
```

    [nltk_data] Downloading package wordnet to /kaggle/working/...
    Archive:  /kaggle/working/corpora/wordnet.zip
       creating: /kaggle/working/corpora/wordnet/
      inflating: /kaggle/working/corpora/wordnet/lexnames  
      inflating: /kaggle/working/corpora/wordnet/data.verb  
      inflating: /kaggle/working/corpora/wordnet/index.adv  
      inflating: /kaggle/working/corpora/wordnet/adv.exc  
      inflating: /kaggle/working/corpora/wordnet/index.verb  
      inflating: /kaggle/working/corpora/wordnet/cntlist.rev  
      inflating: /kaggle/working/corpora/wordnet/data.adj  
      inflating: /kaggle/working/corpora/wordnet/index.adj  
      inflating: /kaggle/working/corpora/wordnet/LICENSE  
      inflating: /kaggle/working/corpora/wordnet/citation.bib  
      inflating: /kaggle/working/corpora/wordnet/noun.exc  
      inflating: /kaggle/working/corpora/wordnet/verb.exc  
      inflating: /kaggle/working/corpora/wordnet/README  
      inflating: /kaggle/working/corpora/wordnet/index.sense  
      inflating: /kaggle/working/corpora/wordnet/data.noun  
      inflating: /kaggle/working/corpora/wordnet/data.adv  
      inflating: /kaggle/working/corpora/wordnet/index.noun  
      inflating: /kaggle/working/corpora/wordnet/adj.exc  



```python
#data loading

train = pd.read_csv('../input/spooky/train.csv')
test = pd.read_csv('../input/spooky/test.csv')
sample = pd.read_csv('../input/spooky/sample_submission.csv')
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
      <th>id</th>
      <th>text</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id26305</td>
      <td>This process, however, afforded me no means of...</td>
      <td>EAP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id17569</td>
      <td>It never once occurred to me that the fumbling...</td>
      <td>HPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id11008</td>
      <td>In his left hand was a gold snuff box, from wh...</td>
      <td>EAP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id27763</td>
      <td>How lovely is spring As we looked from Windsor...</td>
      <td>MWS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id12958</td>
      <td>Finding nothing else, not even gold, the Super...</td>
      <td>HPL</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id02310</td>
      <td>Still, as I urged our leaving Ireland with suc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id24541</td>
      <td>If a fire wanted fanning, it could readily be ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id00134</td>
      <td>And when they had broken down the frail door t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id27757</td>
      <td>While I was thinking how I should possibly man...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id04081</td>
      <td>I am not sure to what limit his knowledge may ...</td>
    </tr>
  </tbody>
</table>
</div>



# Data Cleaning & Exploration


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
      <th>id</th>
      <th>text</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id26305</td>
      <td>This process, however, afforded me no means of...</td>
      <td>EAP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id17569</td>
      <td>It never once occurred to me that the fumbling...</td>
      <td>HPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id11008</td>
      <td>In his left hand was a gold snuff box, from wh...</td>
      <td>EAP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id27763</td>
      <td>How lovely is spring As we looked from Windsor...</td>
      <td>MWS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id12958</td>
      <td>Finding nothing else, not even gold, the Super...</td>
      <td>HPL</td>
    </tr>
  </tbody>
</table>
</div>




```python
#multi-class classification where each sentence is classified into one of 3 classes
train['author'].value_counts().plot(kind = 'bar')
```




    <Axes: xlabel='author'>




    
![png](https://dakshjain97.github.io/assets/img/nlp-glove-lstm_files/nlp-glove-lstm_7_1.png)
    



```python
train.shape
```




    (19579, 3)




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19579 entries, 0 to 19578
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   id      19579 non-null  object
     1   text    19579 non-null  object
     2   author  19579 non-null  object
    dtypes: object(3)
    memory usage: 459.0+ KB



```python
#max & min length of sentences
print(train['text'].str.len().max() , train['text'].str.len().min()) 
```

    4663 21



```python
#distribution of length of sentences follows highly right-skewed dist 
train['text'].str.len().hist(bins = 100)
```




    <Axes: >




    
![png](https://dakshjain97.github.io/assets/img/nlp-glove-lstm_files/nlp-glove-lstm_11_1.png)
    



```python
train['len'] = train['text'].str.len()
```


```python
#only 158 seneteces have >500 length with majority belonging to EAP
train[train['len']>500]['author'].value_counts()
```




    author
    EAP    83
    MWS    53
    HPL    22
    Name: count, dtype: int64




```python
#sample of longest length sentence
train[train['len']==train['len'].max()]['text'].values
```




    array(['Diotima approached the fountain seated herself on a mossy mound near it and her disciples placed themselves on the grass near her Without noticing me who sat close under her she continued her discourse addressing as it happened one or other of her listeners but before I attempt to repeat her words I will describe the chief of these whom she appeared to wish principally to impress One was a woman of about years of age in the full enjoyment of the most exquisite beauty her golden hair floated in ringlets on her shoulders her hazle eyes were shaded by heavy lids and her mouth the lips apart seemed to breathe sensibility But she appeared thoughtful unhappy her cheek was pale she seemed as if accustomed to suffer and as if the lessons she now heard were the only words of wisdom to which she had ever listened The youth beside her had a far different aspect his form was emaciated nearly to a shadow his features were handsome but thin worn his eyes glistened as if animating the visage of decay his forehead was expansive but there was a doubt perplexity in his looks that seemed to say that although he had sought wisdom he had got entangled in some mysterious mazes from which he in vain endeavoured to extricate himself As Diotima spoke his colour went came with quick changes the flexible muscles of his countenance shewed every impression that his mind received he seemed one who in life had studied hard but whose feeble frame sunk beneath the weight of the mere exertion of life the spark of intelligence burned with uncommon strength within him but that of life seemed ever on the eve of fading At present I shall not describe any other of this groupe but with deep attention try to recall in my memory some of the words of Diotima they were words of fire but their path is faintly marked on my recollection It requires a just hand, said she continuing her discourse, to weigh divide the good from evil On the earth they are inextricably entangled and if you would cast away what there appears an evil a multitude of beneficial causes or effects cling to it mock your labour When I was on earth and have walked in a solitary country during the silence of night have beheld the multitude of stars, the soft radiance of the moon reflected on the sea, which was studded by lovely islands When I have felt the soft breeze steal across my cheek as the words of love it has soothed cherished me then my mind seemed almost to quit the body that confined it to the earth with a quick mental sense to mingle with the scene that I hardly saw I felt Then I have exclaimed, oh world how beautiful thou art Oh brightest universe behold thy worshiper spirit of beauty of sympathy which pervades all things, now lifts my soul as with wings, how have you animated the light the breezes Deep inexplicable spirit give me words to express my adoration; my mind is hurried away but with language I cannot tell how I feel thy loveliness Silence or the song of the nightingale the momentary apparition of some bird that flies quietly past all seems animated with thee more than all the deep sky studded with worlds" If the winds roared tore the sea and the dreadful lightnings seemed falling around me still love was mingled with the sacred terror I felt; the majesty of loveliness was deeply impressed on me So also I have felt when I have seen a lovely countenance or heard solemn music or the eloquence of divine wisdom flowing from the lips of one of its worshippers a lovely animal or even the graceful undulations of trees inanimate objects have excited in me the same deep feeling of love beauty; a feeling which while it made me alive eager to seek the cause animator of the scene, yet satisfied me by its very depth as if I had already found the solution to my enquires sic as if in feeling myself a part of the great whole I had found the truth secret of the universe But when retired in my cell I have studied contemplated the various motions and actions in the world the weight of evil has confounded me If I thought of the creation I saw an eternal chain of evil linked one to the other from the great whale who in the sea swallows destroys multitudes the smaller fish that live on him also torment him to madness to the cat whose pleasure it is to torment her prey I saw the whole creation filled with pain each creature seems to exist through the misery of another death havoc is the watchword of the animated world And Man also even in Athens the most civilized spot on the earth what a multitude of mean passions envy, malice a restless desire to depreciate all that was great and good did I see And in the dominions of the great being I saw man reduced?'],
          dtype=object)




```python
#only 195 sentences have <25 length , majority from EAP class
train[train['len']<25]['author'].value_counts()
```




    author
    EAP    101
    MWS     63
    HPL     31
    Name: count, dtype: int64




```python
#sample of shortest sentences
train[train['len']==train['len'].min()]
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
      <th>text</th>
      <th>author</th>
      <th>len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>302</th>
      <td>id20021</td>
      <td>I breathed no longer.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>831</th>
      <td>id18921</td>
      <td>Calderon de la Barca.</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>id16996</td>
      <td>He won't do he won't.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1393</th>
      <td>id12150</td>
      <td>It is still at large.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1704</th>
      <td>id18575</td>
      <td>Do you hear anything?</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2747</th>
      <td>id26561</td>
      <td>My strength was gone.</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2910</th>
      <td>id15628</td>
      <td>My practice was this.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3076</th>
      <td>id21984</td>
      <td>This thought creates.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3799</th>
      <td>id11709</td>
      <td>Did we pass a valley?</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4016</th>
      <td>id17846</td>
      <td>The uproar increases.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4028</th>
      <td>id01048</td>
      <td>My fame is universal.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4059</th>
      <td>id10974</td>
      <td>He reaches the grave.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4241</th>
      <td>id25742</td>
      <td>'Tis berry hebby bug.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5214</th>
      <td>id17463</td>
      <td>Why the third degree?</td>
      <td>HPL</td>
      <td>21</td>
    </tr>
    <tr>
      <th>6034</th>
      <td>id00855</td>
      <td>P. Is not God spirit?</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>6418</th>
      <td>id27928</td>
      <td>"Sunday, the seventh.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>6597</th>
      <td>id13508</td>
      <td>The old man was dead.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>6856</th>
      <td>id07739</td>
      <td>I did so, saying: "M.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>6936</th>
      <td>id22015</td>
      <td>"PIQUANT EXPRESSIONS.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7849</th>
      <td>id21421</td>
      <td>"But how with a gang?</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7991</th>
      <td>id22388</td>
      <td>I am not Suky Snobbs.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>8105</th>
      <td>id18032</td>
      <td>Where was the plague?</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>8474</th>
      <td>id03050</td>
      <td>I gave him love only.</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>8764</th>
      <td>id16503</td>
      <td>He dislikes children.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>9925</th>
      <td>id15121</td>
      <td>but, stay you shall."</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>10254</th>
      <td>id07913</td>
      <td>"How can I move thee?</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>10288</th>
      <td>id03119</td>
      <td>The cards were dealt.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>10825</th>
      <td>id22985</td>
      <td>I cannot rule myself.</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11780</th>
      <td>id05704</td>
      <td>The error is obvious.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>12110</th>
      <td>id13947</td>
      <td>You know what he did.</td>
      <td>HPL</td>
      <td>21</td>
    </tr>
    <tr>
      <th>12553</th>
      <td>id24407</td>
      <td>"One" said the clock.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>12645</th>
      <td>id25967</td>
      <td>In fine, she revived.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>12775</th>
      <td>id23236</td>
      <td>The Duc slips a card.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>13381</th>
      <td>id00223</td>
      <td>You shall be my heir.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>14953</th>
      <td>id14060</td>
      <td>Let me then remember.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>15240</th>
      <td>id08810</td>
      <td>I sickened as I read.</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>15359</th>
      <td>id14135</td>
      <td>Is a native of Spain.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>15518</th>
      <td>id27555</td>
      <td>I leaned to the left.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>15834</th>
      <td>id00377</td>
      <td>We passed St. Paul's.</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>16955</th>
      <td>id09544</td>
      <td>It was dark all dark.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>16957</th>
      <td>id18054</td>
      <td>The populace encored.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>17253</th>
      <td>id14686</td>
      <td>There was the poodle.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>17932</th>
      <td>id18260</td>
      <td>"I forget your arms."</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>18127</th>
      <td>id20376</td>
      <td>Shakspeare's Sonnets.</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
    <tr>
      <th>18518</th>
      <td>id04334</td>
      <td>Women were screaming.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>18741</th>
      <td>id22990</td>
      <td>But it was so silent.</td>
      <td>HPL</td>
      <td>21</td>
    </tr>
    <tr>
      <th>18934</th>
      <td>id04208</td>
      <td>Many were quite awry.</td>
      <td>EAP</td>
      <td>21</td>
    </tr>
    <tr>
      <th>19124</th>
      <td>id08844</td>
      <td>Was my love blamable?</td>
      <td>MWS</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
#high outliers

sns.boxplot(x='author',y='len',data=train)
```




    <Axes: xlabel='author', ylabel='len'>




    
![png](https://dakshjain97.github.io/assets/img/nlp-glove-lstm_files/nlp-glove-lstm_17_1.png)
    



```python
#similar distrbution of sentences length across all classes, HPL having highest median 
plt.figure(figsize=(20,8))
sns.boxplot(x='author',y='len',data=train[train['len']<500])
```




    <Axes: xlabel='author', ylabel='len'>




    
![png](https://dakshjain97.github.io/assets/img/nlp-glove-lstm_files/nlp-glove-lstm_18_1.png)
    



```python
train['len'].describe(percentiles = [0.05,0.25,0.50,0.75,0.95,0.99])
```




    count    19579.000000
    mean       149.057408
    std        106.800189
    min         21.000000
    5%          37.000000
    25%         81.000000
    50%        128.000000
    75%        191.000000
    95%        326.000000
    99%        476.000000
    max       4663.000000
    Name: len, dtype: float64




```python
#finding special characters
ls_sp = sorted([char for char in set(' '.join([word for txt in train['text'] for word in txt])) if re.findall('[^A-Za-z]', char)])
ls_sp
```




    [' ',
     '"',
     "'",
     ',',
     '.',
     ':',
     ';',
     '?',
     'Å',
     'Æ',
     'à',
     'â',
     'ä',
     'æ',
     'ç',
     'è',
     'é',
     'ê',
     'ë',
     'î',
     'ï',
     'ñ',
     'ô',
     'ö',
     'ü',
     'Ν',
     'Ο',
     'Π',
     'Σ',
     'Υ',
     'α',
     'δ',
     'ἶ']




```python
#checking upper cases examples
[sent for sent in list(train['text']) if re.findall('[A-Z]+', sent)]
```




    ['This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.',
     'It never once occurred to me that the fumbling might be a mere mistake.',
     'In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.',
     'How lovely is spring As we looked from Windsor Terrace on the sixteen fertile counties spread beneath, speckled by happy cottages and wealthier towns, all looked as in former years, heart cheering and fair.',
     'Finding nothing else, not even gold, the Superintendent abandoned his attempts; but a perplexed look occasionally steals over his countenance as he sits thinking at his desk.',
     'A youth passed in solitude, my best years spent under your gentle and feminine fosterage, has so refined the groundwork of my character that I cannot overcome an intense distaste to the usual brutality exercised on board ship: I have never believed it to be necessary, and when I heard of a mariner equally noted for his kindliness of heart and the respect and obedience paid to him by his crew, I felt myself peculiarly fortunate in being able to secure his services.',
     'The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.',
     'The surcingle hung in ribands from my body.',
     "I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so.",
     'I confess that neither the structure of languages, nor the code of governments, nor the politics of various states possessed attractions for me.',
     'He shall find that I can feel my injuries; he shall learn to dread my revenge" A few days after he arrived.',
     'Here we barricaded ourselves, and, for the present were secure.',
     'Herbert West needed fresh bodies because his life work was the reanimation of the dead.',
     'The farm like grounds extended back very deeply up the hill, almost to Wheaton Street.',
     'But a glance will show the fallacy of this idea.',
     'He had escaped me, and I must commence a destructive and almost endless journey across the mountainous ices of the ocean, amidst cold that few of the inhabitants could long endure and which I, the native of a genial and sunny climate, could not hope to survive.',
     'To these speeches they gave, of course, their own interpretation; fancying, no doubt, that at all events I should come into possession of vast quantities of ready money; and provided I paid them all I owed, and a trifle more, in consideration of their services, I dare say they cared very little what became of either my soul or my carcass.',
     'Her native sprightliness needed no undue excitement, and her placid heart reposed contented on my love, the well being of her children, and the beauty of surrounding nature.',
     'I even went so far as to speak of a slightly hectic cough with which, at one time, I had been troubled of a chronic rheumatism of a twinge of hereditary gout and, in conclusion, of the disagreeable and inconvenient, but hitherto carefully concealed, weakness of my eyes.',
     "His facial aspect, too, was remarkable for its maturity; for though he shared his mother's and grandfather's chinlessness, his firm and precociously shaped nose united with the expression of his large, dark, almost Latin eyes to give him an air of quasi adulthood and well nigh preternatural intelligence.",
     'Now the net work was not permanently fastened to the hoop, but attached by a series of running loops or nooses.',
     'It was not that the sounds were hideous, for they were not; but that they held vibrations suggesting nothing on this globe of earth, and that at certain intervals they assumed a symphonic quality which I could hardly conceive as produced by one player.',
     'On every hand was a wilderness of balconies, of verandas, of minarets, of shrines, and fantastically carved oriels.',
     'With how deep a spirit of wonder and perplexity was I wont to regard him from our remote pew in the gallery, as, with step solemn and slow, he ascended the pulpit This reverend man, with countenance so demurely benign, with robes so glossy and so clerically flowing, with wig so minutely powdered, so rigid and so vast, could this be he who, of late, with sour visage, and in snuffy habiliments, administered, ferule in hand, the Draconian laws of the academy?',
     'These bizarre attempts at explanation were followed by others equally bizarre.',
     'For many prodigies and signs had taken place, and far and wide, over sea and land, the black wings of the Pestilence were spread abroad.',
     "All that as yet can fairly be said to be known is, that 'Pure gold can be made at will, and very readily from lead in connection with certain other substances, in kind and in proportions, unknown.' Speculation, of course, is busy as to the immediate and ultimate results of this discovery a discovery which few thinking persons will hesitate in referring to an increased interest in the matter of gold generally, by the late developments in California; and this reflection brings us inevitably to another the exceeding inopportuneness of Von Kempelen's analysis.",
     'I seemed to be upon the verge of comprehension without power to comprehend men, at times, find themselves upon the brink of remembrance without being able, in the end, to remember.',
     'Our compasses, depth gauges, and other delicate instruments were ruined; so that henceforth our only reckoning would be guesswork, based on our watches, the calendar, and our apparent drift as judged by any objects we might spy through the portholes or from the conning tower.',
     'This the young warriors took back with them to Sarnath as a symbol of conquest over the old gods and beings of Ib, and a sign of leadership in Mnar.',
     'Meantime the whole Paradise of Arnheim bursts upon the view.',
     'I was rich and young, and had a guardian appointed for me; and all about me would act as if I were one of their great society, while I must keep the secret that I really was cut off from them for ever.',
     'We could make out little by the dim light, but they seemed to contain prophecies, detailed relations of events but lately passed; names, now well known, but of modern date; and often exclamations of exultation or woe, of victory or defeat, were traced on their thin scant pages.',
     'Even now They talked in Their tombs.',
     'Sheehan especially did they ply with inquiries, yet without eliciting any information of value concerning Old Bugs.',
     'He cried aloud once, and a little later gave a gasp that was more terrible than a cry.',
     "The old tracks crossed River Street at grade, and at once veered off into a region increasingly rural and with less and less of Innsmouth's abhorrent fishy odour.",
     'His soul overflowed with ardent affections, and his friendship was of that devoted and wondrous nature that the world minded teach us to look for only in the imagination.',
     'After the first start, he replaced the tissue wrapping around the portrait, as if to shield it from the sordidness of the place.',
     '"The present peculiar condition of affairs at court, and especially of those intrigues in which D is known to be involved, would render the instant availability of the document its susceptibility of being produced at a moment\'s notice a point of nearly equal importance with its possession."',
     "Wilbur's growth was indeed phenomenal, for within three months of his birth he had attained a size and muscular power not usually found in infants under a full year of age.",
     'Pausing, I succeeded with difficulty in raising it, whereupon there was revealed a black aperture, exhaling noxious fumes which caused my torch to sputter, and disclosing in the unsteady glare the top of a flight of stone steps.',
     'It was all mud an\' water, an\' the sky was dark, an\' the rain was wipin\' aout all tracks abaout as fast as could be; but beginnin\' at the glen maouth, whar the trees had moved, they was still some o\' them awful prints big as bar\'ls like he seen Monday."',
     'The visits of Merrival to Windsor, before frequent, had suddenly ceased.',
     'It is not to be supposed, however, that the great Underduk suffered this impertinence on the part of the little old man to pass off with impunity.',
     "I need not tell you how sceptical I have hitherto been on the topic of the soul's immortality.",
     'I often compared myself to them, and finding that my chief superiority consisted in power, I soon persuaded myself that it was in power only that I was inferior to the chiefest potentates of the earth.',
     "And the children's children, and the newcomers' children, grew up.",
     'Dr. Johnson, as I beheld him, was a full, pursy Man, very ill drest, and of slovenly Aspect.',
     'Presently the murmur of water fell gently upon my ear and in a few moments afterward, as I turned with the road somewhat more abruptly than hitherto, I became aware that a building of some kind lay at the foot of a gentle declivity just before me.',
     'Ellison was remarkable in the continuous profusion of good gifts lavished upon him by fortune.',
     'I still continued in the plane of the elipse, but made little progress to the eastward.',
     'It was useless to provide many things, for we should find abundant provision in every town.',
     'They fly quickly over the snow in their sledges; the motion is pleasant, and, in my opinion, far more agreeable than that of an English stagecoach.',
     'I pointed to the spot where he had disappeared, and we followed the track with boats; nets were cast, but in vain.',
     'I have indistinct recollections of a great storm some time after I reached the boat; at any rate, I know that I heard peals of thunder and other tones which Nature utters only in her wildest moods.',
     'There had seemed to be no one in the courtyard below, and I hoped there would be a chance to get away before the spreading of a general alarm.',
     'Mein Gott do you take me vor a shicken?" "No oh no" I replied, much alarmed, "you are no chicken certainly not."',
     'Perpetual fear had jaundiced his complexion, and shrivelled his whole person.',
     'The sun set; the atmosphere grew dim and the evening star no longer shone companionless.',
     'The rain ceased; the clouds sunk behind the horizon; it was now evening, and the sun descended swiftly the western sky.',
     'Nothing, however, occurred except some hill noises; and when the day came there were many who hoped that the new horror had gone as swiftly as it had come.',
     'The state rooms were sufficiently roomy, and each had two berths, one above the other.',
     "It was getting dark, and the ancient roofs and chimney pots outside looked very queer through the bull's eye window panes.",
     'They were marched to various parts of the southern counties, quartered in deserted villages, a part were sent back to their own island, while the season of winter so far revived our energy, that the passes of the country were defended, and any increase of numbers prohibited.',
     'We suffered no little from cold, and the dampness of the atmosphere was most unpleasant; but the ample space in the car enabled us to lie down, and by means of cloaks and a few blankets, we did sufficiently well.',
     'I dared, I conquered them all, till now I have sold myself to death, with the sole condition that thou shouldst follow me Fire, and war, and plague, unite for thy destruction O my Raymond, there is no safety for thee" With an heavy heart I listened to the changes of her delirium; I made her a bed of cloaks; her violence decreased and a clammy dew stood on her brow as the paleness of death succeeded to the crimson of fever, I placed her on the cloaks.',
     'Again he seemed to be in the interior of a house an old house, apparently but the details and inhabitants were constantly changing, and he could never be certain of the faces or the furniture, or even of the room itself, since doors and windows seemed in just as great a state of flux as the more presumably mobile objects.',
     'This gentleman was clothed from head to foot in a richly embroidered black silk velvet pall, wrapped negligently around his form after the fashion of a Spanish cloak.',
     'The pigeons appeared distressed in the extreme, and struggled to escape; while the cat mewed piteously, and, with her tongue hanging out of her mouth, staggered to and fro in the car as if under the influence of poison.',
     '"And what about the window panes?" "They were all gone.',
     'I panted I gasped for breath There could be no doubt of the design of my tormentors oh most unrelenting oh most demoniac of men I shrank from the glowing metal to the centre of the cell.',
     'With great difficulty I gained my feet, and looking dizzily around, was, at first, struck with the idea of our being among breakers; so terrific, beyond the wildest imagination, was the whirlpool of mountainous and foaming ocean within which we were engulfed.',
     'The next morning I delivered my letters of introduction and paid a visit to some of the principal professors.',
     'Nor did Raymond make an end without drawing in vivid and glowing colours, the splendour of a kingdom, in opposition to the commercial spirit of republicanism.',
     'How this celebrated Magazine can sustain its evidently tremendous expenses, is more than we can understand.',
     'In a week or two it had visibly faded, and in the course of a few months it was hardly discernible with the naked eye."',
     'The stranger learned about twenty words at the first lesson; most of them, indeed, were those which I had before understood, but I profited by the others.',
     'Each people looked on the coming struggle as that which would be to a great degree decisive; as, in case of victory, the next step would be the siege of Constantinople by the Greeks.',
     '"But could not the cavity be detected by sounding?"',
     'My lieutenant, for instance, is a man of wonderful courage and enterprise; he is madly desirous of glory, or rather, to word my phrase more characteristically, of advancement in his profession.',
     'Within twenty four hours that machine near the table will generate waves acting on unrecognised sense organs that exist in us as atrophied or rudimentary vestiges.',
     '"Everybody got aout o\' the idee o\' dyin\' excep\' in canoe wars with the other islanders, or as sacrifices to the sea gods daown below, or from snake bite or plague or sharp gallopin\' ailments or somethin\' afore they cud take to the water but simply looked forrad to a kind o\' change that wa\'n\'t a bit horrible arter a while.',
     'The lips were of the usual marble pallor.',
     '"That is absolutely needless," replied G .',
     'Ahead lay sparse grass and scrub blueberry bushes, and beyond them the naked rock of the crag and the thin peak of the dreaded grey cottage.',
     'To be near him, to be loved by him, to feel him again her own, was the limit of her desires.',
     'The sky was serene; and, as I was unable to rest, I resolved to visit the spot where my poor William had been murdered.',
     'I say "thing" be it observed for they tell me the Latin for it is rem.',
     'All that he said threw greatly into the shade Cornelius Agrippa, Albertus Magnus, and Paracelsus, the lords of my imagination; but by some fatality the overthrow of these men disinclined me to pursue my accustomed studies.',
     'The ex queen gives me Idris; Adrian is totally unfitted to succeed to the earldom, and that earldom in my hands becomes a kingdom.',
     'The pupils, too, upon any accession or diminution of light, underwent contraction or dilation, just such as is observed in the feline tribe.',
     '"Keep up the largest branch the one on this side," said Legrand.',
     'And quivering awhile among the draperies of the room, it at length rested in full view upon the surface of the door of brass.',
     'Maternal affection had not rendered Idris selfish; at the beginning of our calamity she had, with thoughtless enthusiasm, devoted herself to the care of the sick and helpless.',
     'He came like a protecting spirit to the poor girl, who committed herself to his care; and after the interment of his friend he conducted her to Geneva and placed her under the protection of a relation.',
     'I had expected some extravagant proposition, and remained silent awhile, collecting my thoughts that I might the better combat her fanciful scheme.',
     'Matters had now assumed a really serious aspect, and I resolved to call at once upon my particular friend, Mr. Theodore Sinivate; for I knew that here at least I should get something like definite information.',
     'The system had its disadvantages, and even its dangers.',
     'Everyone seemed inclined to be silent now, as though holding a secret fear.',
     'They still appeared in public together, and lived under the same roof.',
     '"But, my dear fellow, you are joking then," said I, "this is a very passable skull indeed, I may say that it is a very excellent skull, according to the vulgar notions about such specimens of physiology and your scarabæus must be the queerest scarabæus in the world if it resembles it.',
     'I vaow afur Gawd, I dun\'t know what he wants nor what he\'s a tryin\' to dew."',
     'He began to answer with violence: "Yes, yes, I hate you You are my bane, my poison, my disgust Oh No" And then his manner changed, and fixing his eyes on me with an expression that convulsed every nerve and member of my frame "you are none of all these; you are my light, my only one, my life.',
     'I shivered as I wondered why I did not reach the light, and would have looked down had I dared.',
     'The slow ravages of disease are not pleasant to watch, but in my case there was something subtler and more puzzling in the background.',
     'As the Comte and his associates turned away from the lowly abode of the alchemists, the form of Charles Le Sorcier appeared through the trees.',
     'As I have said, it happened when we were in the medical school, where West had already made himself notorious through his wild theories on the nature of death and the possibility of overcoming it artificially.',
     'Besides, Perdita was with him in his retirement; she saw the moodiness that succeeded to this forced hilarity; she marked his disturbed sleep, his painful irritability once she had seen his tears hers had scarce ceased to flow, since she had beheld the big drops which disappointed pride had caused to gather in his eye, but which pride was unable to dispel.',
     'I packed up my chemical instruments and the materials I had collected, resolving to finish my labours in some obscure nook in the northern highlands of Scotland.',
     'The ideas of my friend may be summed up in a few words.',
     'A definite point among the stars had a claim on him and was calling him.',
     'I believe I shall be forced to call them both out.',
     'God knows what that world can have been, or where he ever glimpsed the blasphemous shapes that loped and trotted and crawled through it; but whatever the baffling source of his images, one thing was plain.',
     'Yet all these appearances have been given I beg pardon will be given by the learned of future ages, to the Ashimah of the Syrians.',
     'My skin was embrowned by the sun; my step was firm with conscious power.',
     'It was in these slums along Main Street that I would find the old Georgian churches, but they were all long abandoned.',
     '"I believe, sir, you have forgotten to pay for your brandy and water."',
     'I started forward and exclaimed, "Villain Before you sign my death warrant, be sure that you are yourself safe."',
     'Only his eyes stayed whole, and they glared with a propulsive, dilated incandescence which grew as the face around them charred and dwindled.',
     'He determined to extract and condense all of glory, power, and achievement, which might have resulted from a long reign, into the three years of his Protectorate.',
     'There was no advertisement of the picking up of this boat.',
     'Baffled, the excavators sought a conference with the Superintendent, who ordered great lengths of rope to be taken to the pit, and spliced and lowered without cessation till a bottom might be discovered.',
     'They had received orders that if I were again taken, I should be brought to the Earl; and his lenity made them expect a conclusion which they considered ill befitting my crime.',
     'Then the lean Nith remarked that no one had seen the old man or his wife since the night the cats were away.',
     'Six years had elapsed, passed in a dream but for one indelible trace, and I stood in the same place where I had last embraced my father before my departure for Ingolstadt.',
     '"But that Kidd\'s accumulations were immense, is well known.',
     'And as I looked, I beheld the head rise, the black, liquid, and deep sunken eyes open in terror, and the thin, shadowed lips part as if for a scream too frightful to be uttered.',
     'And from their high summits, one by one, drop everlasting dews.',
     "I had now spent many hours in tears and mournful meditation; it was past twelve o'clock; all was at peace in the house, and the gentle air that stole in at my window did not rustle the leaves of the twining plants that shadowed it.",
     'With the traditionally receptive ears of the small boy, I learned much; though an habitual secretiveness caused me to tell no one of my information or my resolve.',
     'He did not himself understand these words, or know why certain things made him feel certain emotions; but fancied that some unremembered dream must be responsible.',
     'Look, I weep: for more than two years I have never enjoyed one moment free from anguish.',
     'It is indeed demonstrable that every such impulse given the air, must, in the end, impress every individual thing that exists within the universe; and the being of infinite understanding the being whom we have imagined might trace the remote undulations of the impulse trace them upward and onward in their influences upon all particles of an matter upward and onward for ever in their modifications of old forms or, in other words, in their creation of new until he found them reflected unimpressive at last back from the throne of the Godhead.',
     'Justine was called on for her defence.',
     'His engagement broken, Galpin moved east to begin life anew; but before long, Appletonians heard of his dismissal in disgrace from New York University, where he had obtained an instructorship in English.',
     'He had a narrow head, bulging, watery blue eyes that seemed never to wink, a flat nose, a receding forehead and chin, and singularly undeveloped ears.',
     'It will be remembered, that, in the earliest stage of my speculations upon the possibility of a passage to the moon, the existence, in its vicinity, of an atmosphere, dense in proportion to the bulk of the planet, had entered largely into my calculations; this too in spite of many theories to the contrary, and, it may be added, in spite of a general disbelief in the existence of any lunar atmosphere at all.',
     'My words flowed spontaneously my utterance was firm and quick.',
     'Only a very rare affliction, of course, could bring about such vast and radical anatomical changes in a single individual after maturity changes involving osseous factors as basic as the shape of the skull but then, even this aspect was no more baffling and unheard of than the visible features of the malady as a whole.',
     "As I fled from that accursed castle along the bog's edge I heard a new sound; common, yet unlike any I had heard before at Kilderry.",
     'He did not try to navigate after the first bold flight, for the reaction had taken something out of his soul.',
     'A bear once, attempting to swim from Lofoden to Moskoe, was caught by the stream and borne down, while he roared terribly, so as to be heard on shore.',
     'He obeyed her injunctions, and passed a year in exile in Cumberland.',
     'He reverted to his past life, his successes in Greece, his favour at home.',
     'Murderers, casting it in, would not have failed to attach a weight.',
     'After each short and inevitable sleep I seemed older, whilst my friend aged with a rapidity almost shocking.',
     'I look on the hands which executed the deed; I think on the heart in which the imagination of it was conceived and long for the moment when these hands will meet my eyes, when that imagination will haunt my thoughts no more.',
     'Horrible sights were shaped to me in the turbid cloud that hovered over the city; and my only relief was derived from the struggles I made to approach the gate.',
     'In order to reach it, he said, he would soar through abysses of emptiness, burning every obstacle that stood in his way.',
     '"Pierre Moreau, tobacconist, deposes that he has been in the habit of selling small quantities of tobacco and snuff to Madame L\'Espanaye for nearly four years.',
     'I still quickened my pace.',
     'I resolved to fly far from the scene of my misfortunes; but to me, hated and despised, every country must be equally horrible.',
     'It was in the spirit of this wisdom that, among the ancient Hebrews, it was believed the gates of Heaven would be inevitably opened to that sinner, or saint, who, with good lungs and implicit confidence, should vociferate the word "Amen" It was in the spirit of this wisdom that, when a great plague raged at Athens, and every means had been in vain attempted for its removal, Epimenides, as Laertius relates, in his second book, of that philosopher, advised the erection of a shrine and temple "to the proper God." LYTTLETON BARRY.',
     "I feel confident he never would have dreamed of taking up his residence in Alexander the Great o nopolis had he been aware that, in Alexander the Great o nopolis, there lived a gentleman named John Smith if I rightly remember, who for many years had there quietly grown fat in editing and publishing the 'Alexander the Great o nopolis Gazette.'",
     'I will protect the first the latter I commit to your charge.',
     'Miserable himself that he may render no other wretched, he ought to die.',
     'The police heard a shot in the old Tillinghast house and found us there Tillinghast dead and me unconscious.',
     'It is even possible that the train of my ideas would never have received the fatal impulse that led to my ruin.',
     'I had not entered the army on my own initiative, but rather as a natural result of the enlistment of the man whose indispensable assistant I was the celebrated Boston surgical specialist, Dr. Herbert West.',
     'It was my last link with a chapter of life forever closed, and I valued it highly.',
     'The longer I meditated upon these the more intense grew the interest which had been excited within me.',
     'She often repined; but her trust in the singleness of his affection was undisturbed; and, when they were together, unchecked by fear, she opened her heart to the fullest delight.',
     'To me there was nothing grotesque in the bones and skeletons that strowed some of the stone crypts deep down among the foundations.',
     'He was full of thought, and remained silent during a considerable part of our ride; at length he said, "I must apologize to you for my abstraction; the truth is, Ryland\'s motion comes on to night, and I am considering my reply."',
     'The Thing cannot be described there is no language for such abysms of shrieking and immemorial lunacy, such eldritch contradictions of all matter, force, and cosmic order.',
     'To add to our dilemma, we found the pumps choked and nearly useless.',
     'By these means for they were ignorant men I found little difficulty in gaining them over to my purpose.',
     'It was queer damnably queer and my uncle spoke almost sheepishly, as if half expecting not to be believed, when he declared that of the strange faces many had unmistakably borne the features of the Harris family.',
     'For three months a night has not passed, during the greater part of which I have not been engaged, personally, in ransacking the D Hotel.',
     'Raymond did not wonder, that, thus situated, the artist had shrunk from notice, but he did not for this alter his resolve.',
     'Not any more does he long for the magic of farther hills, or sigh for secrets that peer like green reefs from a bottomless sea.',
     'But while I endured punishment and pain in their defence with the spirit of an hero, I claimed as my reward their praise and obedience.',
     'I must collect my thoughts.',
     'Never imposing upon any one myself, I suffered no one to play the possum with me.',
     '"Not so," said I, "though I confess that my thoughts are not occupied as pleasantly as yours are.',
     'Then, as I remained, paralysed with fear, he found his voice and in his dying breath screamed forth those words which have ever afterward haunted my days and my nights.',
     '"Surely it is not the custom of Englishmen to receive strangers so inhospitably."',
     'They remained confined for five months before the trial took place, the result of which deprived them of their fortune and condemned them to a perpetual exile from their native country.',
     "they never stopped swimmin' in an' aout o' the river from that cursed reef o' Satan an' more an' more attic winders got a boarded up, an' more an' more noises was heerd in haouses as wa'n't s'posed to hev nobody in 'em. . . .",
     'At Lawrence he had been prominent in the mock fraternity of "Tappa Tappa Keg", where he was the wildest and merriest of the wild and merry young roysterers; but this immature, collegiate frivolity did not satisfy him.',
     "The limbs, save for their black fur, roughly resembled the hind legs of prehistoric earth's giant saurians; and terminated in ridgy veined pads that were neither hooves nor claws.",
     'These are my enticements, and they are sufficient to conquer all fear of danger or death and to induce me to commence this laborious voyage with the joy a child feels when he embarks in a little boat, with his holiday mates, on an expedition of discovery up his native river.',
     'There were adequate bolts on the two lateral doors to connecting rooms, and these I proceeded to fasten.',
     'His supposition was that "a well taught boy very thin and tall of his age sufficiently so that he could be concealed in a drawer almost immediately under the chess board" played the game of chess and effected all the evolutions of the Automaton.',
     'Interspersed about the room, crossing and recrossing in endless irregularity, were innumerable benches and desks, black, ancient, and time worn, piled desperately with much bethumbed books, and so beseamed with initial letters, names at full length, grotesque figures, and other multiplied efforts of the knife, as to have entirely lost what little of original form might have been their portion in days long departed.',
     'If manly courage and resistance can save us, we will be saved.',
     'The silver hair and benevolent countenance of the aged cottager won my reverence, while the gentle manners of the girl enticed my love.',
     'I commenced by inuring my body to hardship.',
     'But these absurdities I must not pause to detail.',
     'I travelled only at night, fearful of encountering the visage of a human being.',
     'Scarcely had we recovered our senses, before the foretopsail went into shreds, when we got up a storm stay sail and with this did pretty well for some hours, the ship heading the sea much more steadily than before.',
     'The lower portion of the other is hidden from view by the head of the unwieldy bedstead which is thrust close up against it.',
     'Adrian had introduced systematic modes of proceeding in the metropolis, which, while they were unable to stop the progress of death, yet prevented other evils, vice and folly, from rendering the awful fate of the hour still more tremendous.',
     '"At a quarter past eight, being no longer able to draw breath without the most intolerable pain, I proceeded forthwith to adjust around the car the apparatus belonging to the condenser.',
     "The professor had been stricken whilst returning from the Newport boat; falling suddenly, as witnesses said, after having been jostled by a nautical looking negro who had come from one of the queer dark courts on the precipitous hillside which formed a short cut from the waterfront to the deceased's home in Williams Street.",
     'Suddenly the wretch, animated with his last burst of strength, raised his hideous head from the damp and sunken pavement.',
     '"Ha ha ha" said that gentleman "he he he hi hi hi ho ho ho hu hu hu hu very good indeed You must not be astonished, mon ami; our friend here is a wit a drole you must not understand him to the letter."',
     'It is true that I may not find an opportunity of transmitting it to the world, but I will not fall to make the endeavour.',
     'I repeat that in landscape arrangements, or collocations alone, is the physical Nature susceptible of "exaltation" and that, therefore, her susceptibility of improvement at this one point, was a mystery which, hitherto I had been unable to solve.',
     'He also cut timber and began to repair the unused parts of his house a spacious, peaked roofed affair whose rear end was buried entirely in the rocky hillside, and whose three least ruined ground floor rooms had always been sufficient for himself and his daughter.',
     'In the meantime it was folly to grieve, or to think.',
     'The extreme darkness of the stage, whose only light was received from the fire under the cauldron, joined to a kind of mist that floated about it, rendered the unearthly shapes of the witches obscure and shadowy.',
     "Sheehan's is the acknowledged centre to Chicago's subterranean traffic in liquor and narcotics, and as such has a certain dignity which extends even to the unkempt attachés of the place; but there was until lately one who lay outside the pale of that dignity one who shared the squalor and filth, but not the importance, of Sheehan's.",
     'Perhaps the gradation of his copy rendered it not so readily perceptible; or, more possibly, I owed my security to the master air of the copyist, who, disdaining the letter, which in a painting is all the obtuse can see, gave but the full spirit of his original for my individual contemplation and chagrin.',
     'It was perhaps an effect of such surroundings that my mind early acquired a shade of melancholy.',
     '"You ought to hear, though, what some of the old timers tell about the black reef off the coast Devil Reef, they call it.',
     'The prince had provided all the appliances of pleasure.',
     'For a week I tasted to the full the joys of that charnel conviviality which I must not describe, when the thing happened, and I was borne away to this accursed abode of sorrow and monotony.',
     "He see enough, I tell ye, Mis' Corey This dun't mean no good, an' I think as all the men folks ought to git up a party an' do suthin'.",
     'No recognised school of sculpture had animated this terrible object, yet centuries and even thousands of years seemed recorded in its dim and greenish surface of unplaceable stone.',
     'For a long time I dared not hope; but when his unobstructed breathing and the moisture that suffused his forehead, were tokens no longer to be mistaken of the departure of mortal malady, I ventured to whisper the news of the change to Idris, and at length succeeded in persuading her that I spoke truth.',
     'He comprehended, moreover, the true character, the august aims, the supreme majesty and dignity of the poetic sentiment.',
     'Else there is no immortality for man.',
     'I laid the second tier, and the third, and the fourth; and then I heard the furious vibrations of the chain.',
     'All in all, he became a disconcerting and even gruesome companion; yet in my gratitude for his healing I could not well abandon him to the strangers around him, and was careful to dust his room and attend to his needs each day, muffled in a heavy ulster which I bought especially for the purpose.',
     'I longed for the love which had before filled it to overflowing.',
     'It was a lambent glow of this sort which always played about the old woman and the small furry thing in those lighter, sharper dreams which prefaced his plunge into unknown abysses, and the thought that a wakeful second person could see the dream luminance was utterly beyond sane harbourage.',
     'My first thought was Perdita; to her I must return; her I must support, drawing such food from despair as might best sustain her wounded heart; recalling her from the wild excesses of grief, by the austere laws of duty, and the soft tenderness of regret.',
     'He owned affinity not only with mankind, but all nature was akin to him; the mountains and sky were his friends; the winds of heaven and the offspring of earth his playmates; while he the focus only of this mighty mirror, felt his life mingle with the universe of existence.',
     'Uninquiring souls let this occurrence pass as one of the major clashes in a spasmodic war on liquor.',
     'The rigging was found to be ill fitted, and greatly strained; and on the third day of the blow, about five in the afternoon, our mizzen mast, in a heavy lurch to windward, went by the board.',
     'Alfred stood immoveable before him, his whole attention absorbed.',
     'Beyond it the rusted railway and the Rowley road led off through a flat, marshy terrain dotted with islets of higher and dryer scrub grown land.',
     'Williamson records and traditions were supplied in abundance by my grandfather; though for Orne material I had to depend on my uncle Walter, who put at my disposal the contents of all his files, including notes, letters, cuttings, heirlooms, photographs, and miniatures.',
     'A closer scrutiny, however, satisfied me that it was intended for a kid." "Ha ha" said I, "to be sure I have no right to laugh at you a million and a half of money is too serious a matter for mirth but you are not about to establish a third link in your chain you will not find any especial connexion between your pirates and a goat pirates, you know, have nothing to do with goats; they appertain to the farming interest."',
     'It is no cause for wonder, surely, that even a gang of blackguards should make haste to get home, when a wide river is to be crossed in small boats, when storm impends, and when night approaches.',
     'The connexion of the two events had about it so much of the palpable, that the true wonder would have been a failure of the populace to appreciate and to seize it.',
     'When I heard the fears which had driven the people from Kilderry I laughed as loudly as my friend had laughed, for these fears were of the vaguest, wildest, and most absurd character.',
     'Leave me; I am inexorable."',
     'I listened to my father in silence and remained for some time incapable of offering any reply.',
     'You hate me, but your abhorrence cannot equal that with which I regard myself.',
     'I had been, albeit without definite reason, instinctively on my guard and that was to my advantage in the new and real crisis, whatever it might turn out to be.',
     'I assured my patron that, if this was all, I was perfectly resigned to the task of playing Thomas Hawk.',
     'A large bruise was discovered upon the pit of the stomach, produced, apparently, by the pressure of a knee.',
     "Mem he'll answer, too.",
     'Let a composition be defective, let an emendation be wrought in its mere arrangement of form; let this emendation be submitted to every artist in the world; by each will its necessity be admitted.',
     'The hangman, however, adjusted the noose about my neck.',
     'Then with utter and horrifying suddenness we heard a frightful sound from below.',
     'The boat, however, must not be detained.',
     "We started about four o'clock Legrand, Jupiter, the dog, and myself.",
     "What do maps and records and guide books really tell of the North End? Bah At a guess I'll guarantee to lead you to thirty or forty alleys and networks of alleys north of Prince Street that aren't suspected by ten living beings outside of the foreigners that swarm them.",
     'It would be useless to describe the playing of Erich Zann on that dreadful night.',
     'In this particular instance, it will be understood as most probable, that she proceeded upon a route of more than average diversity from her accustomed ones.',
     'Having noticed these particulars, and some others, I again turned my eyes upon the glorious prospect below, and soon became absorbed in contemplation.',
     'Many of the vessels fired signal guns; and in all we were saluted with loud cheers which we heard with surprising distinctness and the waving of caps and handkerchiefs.',
     'The party spread themselves and hurried from room to room.',
     'He made a vow against love and its train of struggles, disappointment and remorse, and sought in mere sensual enjoyment, a remedy for the injurious inroads of passion.',
     'After that Johansen only brooded over the idol in the cabin and attended to a few matters of food for himself and the laughing maniac by his side.',
     'The latter examined it carefully and deposited it in his pocket book; then, unlocking an escritoire, took thence a letter and gave it to the Prefect.',
     'I retired early and full of dread, and for a long time could not sleep in the uncanny silence of the tower.',
     'I provided myself with a sum of money, together with a few jewels which had belonged to my mother, and departed.',
     'I rushed towards the window, and drawing a pistol from my bosom, fired; but he eluded me, leaped from his station, and running with the swiftness of lightning, plunged into the lake.',
     "Xh, pxh, pxh, Jxhn, dxn't dx sx Yxu've gxt tx gx, yxu knxw, sx gx at xnce, and dxn't gx slxw; fxr nxbxdy xwns yxu here, yxu knxw.",
     'You would have sworn that the writer had been born and brought up in a coffin.',
     '"There are a prodigious number of stately palaces."',
     "She might have heard of my return from London, and my visit to Bolter's Lock, which, connected with my continued absence, might tend greatly to alarm her.",
     'This man, whose name was Beaufort, was of a proud and unbending disposition and could not bear to live in poverty and oblivion in the same country where he had formerly been distinguished for his rank and magnificence.',
     'We found, on each side of the humid pathway, "dry land for the sole of the foot."',
     'I dare not ask you to do what I think right, for I may still be misled by passion.',
     'Refusing to flee, I watched it fade and as I watched I felt that it was in turn watching me greedily with eyes more imaginable than visible.',
     '"Then up with you as soon as possible, for it will soon be too dark to see what we are about."',
     'Since then I must be an object of indifference or contempt to her, better, far better avoid her, nor expose myself before her and the scornful world to the chance of playing the mad game of a fond, foolish Icarus.',
     'I shewed her the dangers which her children incurred during her absence; and she at length agreed not to go beyond the inclosure of the forest.',
     'The entire surface of this metallic enclosure was rudely daubed in all the hideous and repulsive devices to which the charnel superstition of the monks has given rise.',
     'That scene itself must have corresponded to the sealed loft overhead, which had begun to attack his imagination so violently, but later impressions were faint and hazy.',
     '"Mille tonnerres" ejaculated the Prince de Grenouille.',
     'The remains of the half finished creature, whom I had destroyed, lay scattered on the floor, and I almost felt as if I had mangled the living flesh of a human being.',
     'It was plain that the owner had come home; but he had not come from the land, nor from any balloon or airship that could be imagined.',
     "Others thought and still think he'd found an old pirate cache out on Devil Reef.",
     'This is what I saw in the glass: A thin, dark man of medium stature attired in the clerical garb of the Anglican church, apparently about thirty, and with rimless, steel bowed glasses glistening beneath a sallow, olive forehead of abnormal height.',
     'Then again the kindly influence ceased to act I found myself fettered again to grief and indulging in all the misery of reflection.',
     "As the sailor looked in, the gigantic animal had seized Madame L'Espanaye by the hair, which was loose, as she had been combing it, and was flourishing the razor about her face, in imitation of the motions of a barber.",
     'His feelings are forever on the stretch; and when he begins to sink into repose, he finds himself obliged to quit that on which he rests in pleasure for something new, which again engages his attention, and which also he forsakes for other novelties.',
     '"The father of Safie had been the cause of their ruin.',
     'The matter was impossible.',
     '"No" said the Baron, turning abruptly toward the speaker, "dead say you?" "It is indeed true, my lord; and, to a noble of your name, will be, I imagine, no unwelcome intelligence."',
     'At length, one escaping through a sewer, gave freedom to all the rest.',
     '"My children," she said, "my firmest hopes of future happiness were placed on the prospect of your union.',
     'I mean the line Perdidit antiquum litera sonum.',
     "In establishing 'The Tea Pot' he expected to have the field all to himself.",
     'They would have left nothing behind them; for their number would have enabled them to carry all at once.',
     'Time, place, and circumstances rendered it a matter beyond question.',
     'He developed strange caprices, acquiring a fondness for exotic spices and Egyptian incense till his room smelled like the vault of a sepulchred Pharaoh in the Valley of Kings.',
     'They made the special effects, indeed, wrought in the fluid by special impulses, the subject of exact calculation so that it became easy to determine in what precise period an impulse of given extent would engirdle the orb, and impress for ever every atom of the atmosphere circumambient.',
     '"O yes; and for this reason I did not despair.',
     "Haow'd ye like to hear what comes from that awful reef every May Eve an' Hallowmass?",
     'The alienists listened with keen attention to his words, since their curiosity had been aroused to a high pitch by the suggestive yet mostly conflicting and incoherent stories of his family and neighbours.',
     'It must have been midnight at least when Birch decided he could get through the transom.',
     'I will win him to me; he shall not deny his grief to me and when I know his secret then will I pour a balm into his soul and again I shall enjoy the ravishing delight of beholding his smile, and of again seeing his eyes beam if not with pleasure at least with gentle love and thankfulness.',
     'You must be careful and give the thing with a downright improviso air.',
     'As my uncle began slowly and grudgingly to unwrap the things he urged me not to be shocked by the strangeness and frequent hideousness of the designs.',
     'As I looked, a subtle, curious sense of beckoning seemed superadded to the grim repulsion; and oddly enough, I found this overtone more disturbing than the primary impression.',
     'It is curious that although he differed so widely from the mass of Hispanicised and tribal Indians, Romero gave not the least impression of Caucasian blood.',
     'There was Ferdinand Fitz Fossillus Feltspar.',
     'At the termination of this sentence I started, and for a moment, paused; for it appeared to me although I at once concluded that my excited fancy had deceived me it appeared to me that, from some very remote portion of the mansion, there came, indistinctly, to my ears, what might have been, in its exact similarity of character, the echo but a stifled and dull one certainly of the very cracking and ripping sound which Sir Launcelot had so particularly described.',
     'From indications afforded by the barometer, we find that, in ascensions from the surface of the earth we have, at the height of , feet, left below us about one thirtieth of the entire mass of atmospheric air, that at , we have ascended through nearly one third; and that at ,, which is not far from the elevation of Cotopaxi, we have surmounted one half the material, or, at all events, one half the ponderable, body of air incumbent upon our globe.',
     'The task seemed interminable, and I raged almost as violently as the hermit when I saw the hours slipping by in a breathless, foodless round of vain telephoning, and a hectic quest from place to place, hither and thither by subway and surface car.',
     'I soon arrived at the hut: the door was ajar.',
     'Who or what then, was my great great grandmother?',
     'For a moment I was almost paralized by fear; but my energy returned and I demanded a guide to accompany me in following his steps.',
     'I could with pleasure have destroyed the cottage and its inhabitants and have glutted myself with their shrieks and misery.',
     'I breathed no longer.',
     'Nor was any child to be born alive in that house for a century and a half.',
     'Then, as September approached, the clouds began to clear.',
     'I remembered in one of her harshest moments a quotation of mine had roused her to anger and disdain.',
     '"Its adaptation to the eyes which were to behold it upon earth."',
     "At times her screams became insupportable, and for long periods she would utter shrieking horrors which necessitated her son's temporary residence with his cousin, Peleg Harris, in Presbyterian Lane near the new college building.",
     'There had been nothing like order or arrangement.',
     'It was the pictorial carving, however, that did most to hold me spellbound.',
     'The grave was not very deep, but fully as good as that of the previous specimen the thing which had risen of itself and uttered a sound.',
     'About two hours after this occurrence we heard the ground sea, and before night the ice broke and freed our ship.',
     'Accordingly, I was no sooner seated at the card table, with my pretty hostess for a vis à vis, than I propounded those questions the solution of which had become a matter so essential to my peace.',
     'The company followed our example without stint.',
     'I will not pursue these guesses for I have no right to call them more since the shades of reflection upon which they are based are scarcely of sufficient depth to be appreciable by my own intellect, and since I could not pretend to make them intelligible to the understanding of another.',
     'The daughter lay prostrate and motionless; she had swooned.',
     'The abysses were by no means vacant, being crowded with indescribably angled masses of alien hued substance, some of which appeared to be organic while others seemed inorganic.',
     "Past the churchyard, where there were no houses, I could see over the hill's summit and watch the glimmer of stars on the harbour, though the town was invisible in the dark.",
     'Whenever it fell upon me, my blood ran cold; and so by degrees very gradually I made up my mind to take the life of the old man, and thus rid myself of the eye forever.',
     'Upon this occasion we should have been driven out to sea in spite of everything, for the whirlpools threw us round and round so violently, that, at length, we fouled our anchor and dragged it if it had not been that we drifted into one of the innumerable cross currents here to day and gone to morrow which drove us under the lee of Flimen, where, by good luck, we brought up.',
     'Disappointment and sickness have hitherto held dominion over me; twin born with me, my would, was for ever enchained by the shall not, of these my tyrants.',
     'The perfume in my nostrils died away.',
     'Adrian led the troops.',
     'The tension on our part became very great.',
     'Yet she smiled on and still on, uncomplainingly, because she saw that the painter who had high renown took a fervid and burning pleasure in his task, and wrought day and night to depict her who so loved him, yet who grew daily more dispirited and weak.',
     'You on earth have unwittingly felt its distant presence you who without knowing idly gave to its blinking beacon the name of Algol, the Daemon Star.',
     'Again a furtive trying of a bolted connecting door, and again a receding creaking.',
     'The sun arose with a sickly yellow lustre, and clambered a very few degrees above the horizon emitting no decisive light.',
     'The weather was cold; and, upon quitting my own room, I had thrown a cloak over my dressing wrapper, putting it off upon reaching the scene of play.',
     '"How much was the reward offered, did you say?" asked Dupin.',
     'I was still alone, for as much as I feared the unknown horror I sought, there was more fear in the thought of telling anybody.',
     'I do not weep or sigh; but I must reason with myself, and force myself to feel sorrow and despair.',
     'Our conversation was now long, earnest, uninterrupted, and totally unreserved.',
     "He knew things he didn't dare put into that stupid Magnalia or that puerile Wonders of the Invisible World.",
     'Briden looked back and went mad, laughing shrilly as he kept on laughing at intervals till death found him one night in the cabin whilst Johansen was wandering deliriously.',
     'Moreover, my occupations have been always made to chime in with the ordinary habitudes of my fellowmen.',
     'To this man Iranon spoke, as to so many others: "Canst thou tell me where I may find Aira, the city of marble and beryl, where flows the hyaline Nithra and where the falls of the tiny Kra sing to verdant valleys and hills forested with yath trees?"',
     '"Were I misanthropic," he said, "such a locale would suit me.',
     'In my childhood the shunned house was vacant, with barren, gnarled, and terrible old trees, long, queerly pale grass, and nightmarishly misshapen weeds in the high terraced yard where birds never lingered.',
     'He found my name a good passport to preferment, and he had procured for me the situation of private secretary to the Ambassador at Vienna, where I should enter on my career under the best auspices.',
     "There were reasons why I would have been glad to let the war separate us; reasons why I found the practice of medicine and the companionship of West more and more irritating; but when he had gone to Ottawa and through a colleague's influence secured a medical commission as Major, I could not resist the imperious persuasion of one determined that I should accompany him in my usual capacity.",
     'But on you only had I any claim for pity and redress, and from you I determined to seek that justice which I vainly attempted to gain from any other being that wore the human form.',
     'I strove to think that all this grandeur was but more glaring infamy, and that, by planting his gold enwoven flag beside my tarnished and tattered banner, he proclaimed not his superiority, but his debasement.',
     'I will not dwell longer than I need on these disastrous circumstances.',
     'Geographically it lay about two miles northwest of the base of Tempest Mountain, and three miles from the oak girt mansion.',
     '"D ," replied Dupin, "is a desperate man, and a man of nerve.',
     '"Upon honor," said I. "Nose and all?" she asked.',
     'I heard the next morning from the steward that upon his arrival he had been in a most terrible state of mind: he had passed the first night in the garden lying on the damp grass; he did not sleep but groaned perpetually.',
     'Then the lock of the connecting door to my room was softly tried.',
     'This had probably been done with the view of arousing me from sleep.',
     'It was at this moment that his eyes, and mine also, fell upon the scrap of parchment, which I then supposed to be paper.',
     'On his throat were the marks of murderous hands, and on his left ankle was a distressing rat bite.',
     'Everything is related in them which bears reference to my accursed origin; the whole detail of that series of disgusting circumstances which produced it is set in view; the minutest description of my odious and loathsome person is given, in language which painted your own horrors and rendered mine indelible.',
     'When we reached this tree, Legrand turned to Jupiter, and asked him if he thought he could climb it.',
     'I gave him pretty nearly the same account of my former pursuits as I had given to his fellow professor.',
     'Thus was I left to ponder on myself as the only human creature within the great fortress, and in my utter solitude my mind began to cease its vain protest against the impending doom, to become almost reconciled to the fate which so many of my ancestors had met.',
     'As the music went on, my ideas seemed to quit their mortal dwelling house; they shook their pinions and began a flight, sailing on the placid current of thought, filling the creation with new glory, and rousing sublime imagery that else had slept voiceless.',
     'I waited for my letters with feverish impatience; if they were delayed I was miserable and overcome by a thousand fears; and when they arrived and I saw the superscription of Elizabeth or my father, I hardly dared to read and ascertain my fate.',
     'It is now madness or hypocrisy to doubt.',
     'But even in this acute moment my chief horror was something apart from the immediate weakness of my defences.',
     "Trade fallin' off, mills losin' business even the new ones an' the best of our menfolks kilt a privateerin' in the War of or lost with the Elizy brig an' the Ranger snow both of 'em Gilman venters.",
     'One fine February day, when the sun had reassumed some of its genial power, I walked in the forest with my family.',
     'Ingenuity: Your diddler is ingenious.',
     "all over the reef an' swimmin' up the harbour into the Manuxet. . . .",
     'At this time Lord Raymond returned from Greece.',
     'Bread and water did not tame my blood, nor solitary confinement inspire me with gentle thoughts.',
     "The moon's distance from the earth is, in round numbers, , miles.",
     'I carried pistols and a dagger constantly about me and was ever on the watch to prevent artifice, and by these means gained a greater degree of tranquillity.',
     'This accident, with the loss of my insurance, and with the more serious loss of my hair, the whole of which had been singed off by the fire, predisposed me to serious impressions, so that, finally, I made up my mind to take a wife.',
     'As these crossed the direct line of my vision they affected me as forms; but upon passing to my side their images impressed me with the idea of shrieks, groans, and other dismal expressions of terror, of horror, or of wo.',
     'So entirely secluded, and in fact inaccessible, except through a series of accidents, is the entrance of the ravine, that it is by no means impossible that I was indeed the first adventurer the very first and sole adventurer who had ever penetrated its recesses.',
     'In ceasing, he departed at once, and as abruptly as he had entered.',
     'We could not understand, that is to say, we could not have understood, had the notion of this primum mobile ever obtruded itself; we could not have understood in what manner it might be made to further the objects of humanity, either temporal or eternal.',
     'The event, under the circumstances, was tremendous; for in the existence of a subterranean space here, my mad theories had terrible confirmation.',
     'The gentle words of Agatha and the animated smiles of the charming Arabian were not for me.',
     'Alas Life is obstinate and clings closest where it is most hated.',
     'The voice, however, still puzzled me no little; but even this apparent mystery was speedily cleared up.',
     "Gilman's dreams consisted largely in plunges through limitless abysses of inexplicably coloured twilight and bafflingly disordered sound; abysses whose material and gravitational properties, and whose relation to his own entity, he could not even begin to explain.",
     'I was struck by the improvement that appeared in the health of Adrian.',
     '"Are you better now, sir?" said she.',
     'Another woman confirmed the account of the fishermen having brought the body into her house; it was not cold.',
     'I could now find room to doubt the evidence of my senses; and seldom called up the subject at all but with wonder at extent of human credulity, and a smile at the vivid force of the imagination which I hereditarily possessed.',
     'His appearance, different from any I had ever before seen, and his flight somewhat surprised me.',
     'I am thy creature, and I will be even mild and docile to my natural lord and king if thou wilt also perform thy part, the which thou owest me.',
     'The insane yarn I had heard from the aged drunkard did not promise very pleasant dreams, and I felt I must keep the image of his wild, watery eyes as far as possible from my imagination.',
     'Here, divesting myself of my clothes, for there is no reason why we cannot die as we were born, I threw myself headlong into the current; the sole witness of my fate being a solitary crow that had been seduced into the eating of brandy saturated corn, and so had staggered away from his fellows.',
     'At a wave of my hand my deliverers hurried tumultuously away.',
     'You should have seen me you should.',
     'And, in the midst of all this, the continuous braying of a donkey arose over all.',
     'There was none of the exotic technique you see in Sidney Sime, none of the trans Saturnian landscapes and lunar fungi that Clark Ashton Smith uses to freeze the blood.',
     'For Arthur Munroe was dead.',
     'Nor did I doubt, that in the tranquillity of our family circle at Windsor, she would recover some degree of composure, and in the end, of happiness.',
     'These Great Old Ones, Castro continued, were not composed altogether of flesh and blood.',
     'Having thus fathomed, to his satisfaction, the intentions of Jehovah, out of these intentions he built his innumerable systems of mind.',
     'No man could crush a "butterfly on the wheel" with better effect; no man better cover a speedy retreat from a powerful adversary.',
     'In fact, to write upon such a theme it is necessary to have been hanged.',
     'Their still apparent union permitted her to do much; but no woman could, in the end, present a remedy to the encreasing negligence of the Protector; who, as if seized with a paroxysm of insanity, trampled on all ceremony, all order, all duty, and gave himself up to license.',
     "When the old woman began to turn toward him he fled precipitately off the bridge and into the shelter of the town's labyrinthine waterfront alleys.",
     'The rooks cawed loudly in the trees above; mixed with their hoarse cries I heard a lively strain of music.',
     'I took it because of the queer old brick well in the cellar one of the sort I told you about.',
     'No sympathy may I ever find.',
     'Steeped in misery as I am misery, alas only too real I shall be pardoned for seeking relief, however slight and temporary, in the weakness of a few rambling details.',
     'I suppose some astonishment was exhibited in my countenance, for Mr. Kirwin hastened to say, "Immediately upon your being taken ill, all the papers that were on your person were brought me, and I examined them that I might discover some trace by which I could send to your relations an account of your misfortune and illness.',
     'He knew deeper vices through books, and he now longed to know them at first hand.',
     'As soon as morning dawned I crept from my kennel, that I might view the adjacent cottage and discover if I could remain in the habitation I had found.',
     '"Ass" said the fourth.',
     'Elinor die This is frenzy and the most miserable despair: you cannot die while I am near."',
     "Then too, we had spoken to him in advance of our quest; and I felt after my uncle's going that he would understand and assist me in some vitally necessary public explanations.",
     "Our talk was on subjects, unconnected with the emotions that evidently occupied each; but we each divined the other's thought, and as our voices spoke of indifferent matters, our eyes, in mute language, told a thousand things no tongue could have uttered.",
     'I was very sorry for him, for I dislike to see a German suffer; but he was not a good man to die with.',
     "An' the smell was awful, like what it is araound Wizard Whateley's ol' haouse. . .",
     'Nor did Adrian instruct me only in the cold truths of history and philosophy.',
     'This relieves us of all doubt upon the question whether the old lady could have first destroyed the daughter and afterward have committed suicide.',
     'On a certain rainy afternoon when this illusion seemed phenomenally strong, and when, in addition, I had fancied I glimpsed a kind of thin, yellowish, shimmering exhalation rising from the nitrous pattern toward the yawning fireplace, I spoke to my uncle about the matter.',
     "'Let me go,' he cried; 'monster Ugly wretch You wish to eat me and tear me to pieces.",
     "In the open air alone I found relief; among nature's beauteous works, her God reassumed his attribute of benevolence, and again I could trust that he who built up the mountains, planted the forests, and poured out the rivers, would erect another state for lost humanity, where we might awaken again to our affections, our happiness, and our faith.",
     'Accordingly I kept north along Main to Martin, then turning inland, crossing Federal Street safely north of the Green, and entering the decayed patrician neighbourhood of northern Broad, Washington, Lafayette, and Adams Streets.',
     'In Bolton the prevailing spirit of Puritanism had outlawed the sport of boxing with the usual result.',
     'I went down to open it with a light heart, for what had I now to fear?',
     'I reason a priori, and a diddle would be no diddle without a grin.',
     '"Vase am I," she thought, "vase brimful of despair\'s direst essence.',
     'He alluded to Adrian, and spoke of him with that disparagement that the worldly wise always attach to enthusiasm.',
     'I recollected my threat and resolved that it should be accomplished.',
     "Raymond had evidently vacillated during his journey, and irresolution was marked in every gesture as we entered Perdita's cottage.",
     'She spoke no word; and I not for worlds could I have uttered a syllable.',
     'This is somewhat thick, and so are her ankles, but she has a fine pair of green stockings to cover them.',
     'Thus Mark Antony composed a treatise upon getting drunk.',
     'Yet where had the fellow got such an odd notion?',
     'At night he would not be alone, nor would the company of a few persons calm him.',
     'P. I do not comprehend.',
     'When my father had arrived the storm had already begun, but he had refused to stop and leaving his horse there he walked on towards the sea.',
     'Its persistence among a simple people was quite natural in view of the sudden and permanent return of abundantly fine fishing, and it soon came to be the greatest influence on the town, replacing Freemasonry altogether and taking up headquarters in the old Masonic Hall on New Church Green.',
     'We both inserted the whole unopened wooden box, closed the door, and started the electricity.',
     'The islands were no longer visible; whether they had passed down the horizon to the southeast, or whether my increasing elevation had left them out of sight, it is impossible to say.',
     'His dreams were meanwhile increasing in vividness, and though shewing him none of the strange cities and incredible gardens of the old days, were assuming a definite cast whose purpose could not be mistaken.',
     'Alas, what will become of us?',
     '"The relatives of the Earl of Windsor," said she haughtily, "doubtless think that I injured him; perhaps the Earl himself would be the first to acquit me, but probably I do not deserve acquittal.',
     'Now, as the baying of that dead, fleshless monstrosity grows louder and louder, and the stealthy whirring and flapping of those accursed web wings circles closer and closer, I shall seek with my revolver the oblivion which is my only refuge from the unnamed and unnamable.',
     "If we discover repetitions of such letters, so arranged, they will most probably represent the word 'the.' Upon inspection, we find no less than seven such arrangements, the characters being;.",
     'We careered round and round for perhaps an hour, flying rather than floating, getting gradually more and more into the middle of the surge, and then nearer and nearer to its horrible inner edge.',
     'His demeanour was sad; for a moment he appeared glad to see me and then he checked himself as if unwilling to betray his feelings.',
     'The wound, and consequent inability of Argyropylo, caused Raymond to be the first in command.',
     'At length we pulled away.',
     'But it is not that the corpse was found to have the garters of the missing girl, or found to have her shoes, or her bonnet, or the flowers of her bonnet, or her feet, or a peculiar mark upon the arm, or her general size and appearance it is that the corpse had each, and all collectively.',
     "Dr. West's reanimated specimens were not meant for long existence or a large audience.",
     'I often wished that I had permitted her to take her own course, and engage herself in such labours for the welfare of others as might have distracted her thoughts.',
     'We lowered it without difficulty, although it was only by a miracle that we prevented it from swamping as it touched the water.',
     'Did not proceed up stairs.',
     'The old woman always appeared out of thin air near the corner where the downward slant met the inward slant.',
     'But it was not in these vortices of complete alienage that he saw Brown Jenkin.',
     'Now, beware Be silent Do not urge me to your destruction.',
     'The open book lay flat between us, with the picture staring repulsively upward.',
     'I motioned to him to continue, which he did with renewed signs of reluctance.',
     'We went out with the resolution of disputing with our foe.',
     ", it will be seen at pp. and , that this illustrious chemist had not only conceived the idea now in question, but had actually made no inconsiderable progress, experimentally, in the very identical analysis now so triumphantly brought to an issue by Von Kempelen, who although he makes not the slightest allusion to it, is, without doubt I say it unhesitatingly, and can prove it, if required, indebted to the 'Diary' for at least the first hint of his own undertaking.",
     'He, Valence, knew Marie, and could not be mistaken in her identity.',
     'We will fight the enemy to the last.',
     '"I repeat, then, that I only half felt, and never intellectually believed.',
     'He was merely crass of fibre and function thoughtless, careless, and liquorish, as his easily avoidable accident proves, and without that modicum of imagination which holds the average citizen within certain limits fixed by taste.',
     'Will nobody contrive a more expeditious mode of progress?',
     'I spoke without much heed, and the very conclusion of what I said brought with it other thoughts.',
     'This exception was found in the person of a scholar, who, although no relation, bore the same Christian and surname as myself; a circumstance, in fact, little remarkable; for, notwithstanding a noble descent, mine was one of those everyday appellations which seem, by prescriptive right, to have been, time out of mind, the common property of the mob.',
     'She reposed beside her beloved, and the tomb above was inscribed with the united names of Raymond and Perdita.',
     'What this meant, no one could quite be certain till later.',
     'Her story was plain to him, plain and distinct as the remorse and horror that darted their fangs into him.',
     'I was also sorry that I had no one with whom to converse.',
     'What I saw unnerved me most surprisingly, considering its relative triviality.',
     'SIR, Through our common friend, Mr. P., I have received your note of this evening.',
     'If I wept he would gaze on me in silence but he was no longer harsh and although he repulsed every caress yet it was with gentleness.',
     'It is indeed early," he continued, musingly, as a cherub with a heavy golden hammer made the apartment ring with the first hour after sunrise: "It is indeed early but what matters it?',
     "These sounds were mingled with the roaring of the sea, the splash of the chafed billows round the vessel's sides, and the gurgling up of the water in the hold.",
     'I had entered the brush grown cut and was struggling along at a very slow pace when that damnable fishy odour again waxed dominant.',
     'But I could discover, amidst all her repinings, deep resentment towards Raymond, and an unfading sense of injury, that plucked from me my hope, when I appeared nearest to its fulfilment.',
     'After all, what is it?',
     'He rode through the town, visiting the wounded, and giving such orders as were necessary for the siege he meditated.',
     'This box was three feet and a half long, three feet broad, and two and a half feet deep.',
     'My every motion was undoubtedly watched.',
     'After that experience West had dropped his researches for some time; but as the zeal of the born scientist slowly returned, he again became importunate with the college faculty, pleading for the use of the dissecting room and of fresh human specimens for the work he regarded as so overwhelmingly important.',
     'Eternal night continued to envelop us, all unrelieved by the phosphoric sea brilliancy to which we had been accustomed in the tropics.',
     '"Shocking" said the youth, calmly, and turned quietly into the chateau.',
     'This was not altogether the fact: but predominant self will assumed the arms and masque of callous feeling; and the haughty lady disdained to exhibit any token of the struggle she endured; while the slave of pride, she fancied that she sacrificed her happiness to immutable principle.',
     'The police have laid bare the floors, the ceilings, and the masonry of the walls, in every direction.',
     'Uprearing themselves in tall slender lines of light, they thus remained burning all pallid and motionless; and in the mirror which their lustre formed upon the round table of ebony at which we sat, each of us there assembled beheld the pallor of his own countenance, and the unquiet glare in the downcast eyes of his companions.',
     'Evidently the ticket agent had not exaggerated the dislike which local people bore toward Innsmouth and its denizens.',
     'I was now in Holborn, and passed by a public house filled with uproarious companions, whose songs, laughter, and shouts were more sorrowful than the pale looks and silence of the mourner.',
     'If I were awake I should like to die, but now it is no matter.',
     'I rambled on, oppressed, distracted by painful emotions suddenly I found myself before Drury Lane Theatre.',
     'He supposes that, had this been the case, it might have appeared at the surface on the Wednesday, and thinks that only under such circumstances it could so have appeared.',
     'Then came the frenzied tones again: "Carter, it\'s terrible monstrous unbelievable" This time my voice did not fail me, and I poured into the transmitter a flood of excited questions.',
     'Excellent friend how sincerely you did love me, and endeavour to elevate my mind until it was on a level with your own.',
     'While in London these and many other dreadful thoughts too harrowing for words were my portion: I lost all this suffering when I was free; when I saw the wild heath around me, and the evening star in the west, then I could weep, gently weep, and be at peace.',
     "And yet, as the members severally shook their heads and confessed defeat at the Inspector's problem, there was one man in that gathering who suspected a touch of bizarre familiarity in the monstrous shape and writing, and who presently told with some diffidence of the odd trifle he knew.",
     '"What can I do?" she cried, "I am lost we are both for ever lost But come come with me, Lionel; here I must not stay, we can get a chaise at the nearest post house; yet perhaps we have time come, O come with me to save and protect me" When I heard her piteous demands, while with disordered dress, dishevelled hair, and aghast looks, she wrung her hands the idea shot across me is she also mad? "Sweet one," and I folded her to my heart, "better repose than wander further; rest my beloved, I will make a fire you are chill."',
     'My productions however were sufficiently unpretending; they were confined to the biography of favourite historical characters, especially those whom I believed to have been traduced, or about whom clung obscurity and doubt.',
     'But I was enchanted by the appearance of the hut; here the snow and rain could not penetrate; the ground was dry; and it presented to me then as exquisite and divine a retreat as Pandemonium appeared to the demons of hell after their sufferings in the lake of fire.',
     'And it was under a horned waning moon that I saw the city for the first time.',
     'I determined to follow my nose.',
     'She received these tokens of returning love with gentleness; she did not shun his company; but she endeavoured to place a barrier in the way of familiar intercourse or painful discussion, which mingled pride and shame prevented Raymond from surmounting.',
     'Within the wall thus exposed by the displacing of the bones, we perceived a still interior recess, in depth about four feet, in width three, in height six or seven.',
     'Here were the future governors of England; the men, who, when our ardour was cold, and our projects completed or destroyed for ever, when, our drama acted, we doffed the garb of the hour, and assumed the uniform of age, or of more equalizing death; here were the beings who were to carry on the vast machine of society; here were the lovers, husbands, fathers; here the landlord, the politician, the soldier; some fancied that they were even now ready to appear on the stage, eager to make one among the dramatis personae of active life.',
     'Perhaps it is madness that is overtaking me yet perhaps a greater horror or a greater marvel is reaching out.',
     'This extended but little below the elbow.',
     'Here, truly, was the apotheosis of the unnamable.',
     'He had been saying to himself "It is nothing but the wind in the chimney it is only a mouse crossing the floor," or "It is merely a cricket which has made a single chirp."',
     'Thus it seemed to me evident that my rate of ascent was not only on the increase, but that the progression would have been apparent in a slight degree even had I not discharged the ballast which I did.',
     'A low, continuous murmur, like that arising from a full, but gently flowing river, came to my ears, intermingled with the peculiar hum of multitudinous human voices.',
     'I repeat that the principle here expressed, is incontrovertible; but there may be something even beyond it.',
     "Evadne, once the idol of Adrian's affections; and who, for the sake of her present visitor, had disdained the noble youth, and then, neglected by him she loved, with crushed hopes and a stinging sense of misery, had returned to her native Greece.",
     'By observing, as I say, the strictest system in all my dealings, and keeping a well regulated set of books, I was enabled to get over many serious difficulties, and, in the end, to establish myself very decently in the profession.',
     'But I was bewildered, perplexed, and unable to arrange my ideas sufficiently to understand the full extent of his proposition.',
     'But this thought, which supported me in the commencement of my career, now serves only to plunge me lower in the dust.',
     'Great God If for one instant I had thought what might be the hellish intention of my fiendish adversary, I would rather have banished myself forever from my native country and wandered a friendless outcast over the earth than have consented to this miserable marriage.',
     'Those who have watched the tall, lean, Terrible Old Man in these peculiar conversations, do not watch him again.',
     'This idea, having once seized upon my fancy, greatly excited it, and I lost myself forthwith in revery.',
     'Holding a lighted candle at this door, and shifting the position of the whole machine repeatedly at the same time, a bright light is thrown entirely through the cupboard, which is now clearly seen to be full, completely full, of machinery.',
     'The apartments of the students were converted into so many pot houses, and there was no pot house of them all more famous or more frequented than that of the Baron.',
     'West, young despite his marvellous scientific acquirements, had scant patience with good Dr. Halsey and his erudite colleagues; and nursed an increasing resentment, coupled with a desire to prove his theories to these obtuse worthies in some striking and dramatic fashion.',
     'Valdemar, do you still sleep?"',
     'Some time before noon his physician, Dr. Hartwell, called to see him and insisted that he cease work.',
     'In this manner I distributed my occupations when I first arrived, but as I proceeded in my labour, it became every day more horrible and irksome to me.',
     'Such a monster has, then, really existence I cannot doubt it, yet I am lost in surprise and admiration.',
     'By the operation of this spring, the screw is made to revolve with great rapidity, communicating a progressive motion to the whole.',
     'The scent had never for an instant been lost.',
     'Throwing the links about his waist, it was but the work of a few seconds to secure it.',
     'It was open wide, wide open and I grew furious as I gazed upon it.',
     'Gilman could not be very clear about his reasons for this last assumption, but his haziness here was more than overbalanced by his clearness on other complex points.',
     'You should have seen how wisely I proceeded with what caution with what foresight with what dissimulation I went to work I was never kinder to the old man than during the whole week before I killed him.',
     'It is hard for one so young who was once so happy as I was; sic voluntarily to divest themselves of all sensation and to go alone to the dreary grave; I dare not.',
     'The aspect of that ruin I cannot describe I must have been mad, for it seemed to rise majestic and undecayed, splendid and column cinctured, the flame reflecting marble of its entablature piercing the sky like the apex of a temple on a mountain top.',
     'And when did the radiant Una ask anything of her Monos in vain?',
     'In her resentful mood, these expressions had been remembered with acrimony and disdain; they visited her in her softened hour, taking sleep from her eyes, all hope of rest from her uneasy mind.',
     'God raises my weakness and gives me courage to endure the worst.',
     'Windows there were none.',
     'His own face was in shadow, and he wore a wide brimmed hat which somehow blended perfectly with the out of date cloak he affected; but I was subtly disquieted even before he addressed me.',
     'What other possible reason could there have been for her so blushing?',
     'I could not have been more surprised at the sound of the trump of the Archangel.',
     'And here how singularly sounds that word which of old was wont to bring terror to all hearts throwing a mildew upon all pleasures Una.',
     'Thinking earnestly upon these points, I remained, for an hour perhaps, half sitting, half reclining, with my vision riveted upon the portrait.',
     'Upon my word of honor, this was not an unreasonable price for that dickey.',
     'Give me the name of friend; I will fulfill its duties; and if for a moment complaint and sorrow would shape themselves into words let me be near to speak peace to your vext soul."',
     'The glare from the enkindled roof illumined its inmost recesses.',
     'The king has ordered some novel spectacle some gladiatorial exhibition at the hippodrome or perhaps the massacre of the Scythian prisoners or the conflagration of his new palace or the tearing down of a handsome temple or, indeed, a bonfire of a few Jews.',
     'I might as well have attempted to arrest an avalanche Down still unceasingly still inevitably down I gasped and struggled at each vibration.',
     'They who dream by day are cognizant of many things which escape those who dream only by night.',
     'Long and earnestly did I continue the investigation: but the contemptible reward of my industry and perseverance proved to be only a set of false teeth, two pair of hips, an eye, and a bundle of billets doux from Mr. Windenough to my wife.',
     'After great trouble, occasioned by the intractable ferocity of his captive during the home voyage, he at length succeeded in lodging it safely at his own residence in Paris, where, not to attract toward himself the unpleasant curiosity of his neighbors, he kept it carefully secluded, until such time as it should recover from a wound in the foot, received from a splinter on board ship.',
     'It was an odd scene, and because I was strange to New England I had never known its like before.',
     'It was a key a guide to certain gateways and transitions of which mystics have dreamed and whispered since the race was young, and which lead to freedoms and discoveries beyond the three dimensions and realms of life and matter that we know.',
     'As the hour for its arrival drew near I noticed a general drift of the loungers to other places up the street, or to the Ideal Lunch across the square.',
     'It seems to me rather merciful that I do not, for they were terrible studies, which I pursued more through reluctant fascination than through actual inclination.',
     'Her manners were cold and repulsive.',
     'She is very clever and gentle, and extremely pretty; as I mentioned before, her mien and her expression continually remind me of my dear aunt.',
     'Soon after we heard that the poor victim had expressed a desire to see my cousin.',
     'Looking about, I saw that the ceiling was wet and dripping; the soaking apparently proceeding from a corner on the side toward the street.',
     'Nitrogen, on the contrary, was incapable of supporting either animal life or flame.',
     'We read of centre and wing in Greek and Roman history; we fancy a spot, plain as a table, and soldiers small as chessmen; and drawn forth, so that the most ignorant of the game can discover science and order in the disposition of the forces.',
     'Engaged in researches which absorbed our whole attention, it had been nearly a month since either of us had gone abroad, or received a visiter, or more than glanced at the leading political articles in one of the daily papers.',
     'Whisper it not, let the demons hear and rejoice The choice is with us; let us will it, and our habitation becomes a paradise.',
     'Her lower tones were absolutely miraculous.',
     '"On Monday, one of the bargemen connected with the revenue service, saw a empty boat floating down the Seine.',
     'I am a farce and play to him, but to me this is all dreary reality: he takes all the profit and I bear all the burthen.',
     'We have already reached a very high latitude; but it is the height of summer, and although not so warm as in England, the southern gales, which blow us speedily towards those shores which I so ardently desire to attain, breathe a degree of renovating warmth which I had not expected.',
     'He was nearly killed with kindness.',
     'One afternoon there was a discussion of possible freakish curvatures in space, and of theoretical points of approach or even contact between our part of the cosmos and various other regions as distant as the farthest stars or the trans galactic gulfs themselves or even as fabulously remote as the tentatively conceivable cosmic units beyond the whole Einsteinian space time continuum.',
     'I listened to his statement, which was delivered without any presumption or affectation, and then added that his lecture had removed my prejudices against modern chemists; I expressed myself in measured terms, with the modesty and deference due from a youth to his instructor, without letting escape inexperience in life would have made me ashamed any of the enthusiasm which stimulated my intended labours.',
     'For example, I bought pen, ink and paper, and put them into furious activity.',
     'Who, indeed, among my most abandoned associates, would not rather have disputed the clearest evidence of his senses, than have suspected of such courses, the gay, the frank, the generous William Wilson the noblest and most liberal commoner at Oxford him whose follies said his parasites were but the follies of youth and unbridled fancy whose errors but inimitable whim whose darkest vice but a careless and dashing extravagance?',
     'I have consorted long with grief, entered the gloomy labyrinth of madness, and emerged, but half alive.',
     'Hereupon I bethought me of looking immediately before my nose, and there, sure enough, confronting me at the table sat a personage nondescript, although not altogether indescribable.',
     'I dared not make the effort which was to satisfy me of my fate and yet there was something at my heart which whispered me it was sure.',
     'Certainly it sounds absurd to hear that a woman educated only in the rudiments of French often shouted for hours in a coarse and idiomatic form of that language, or that the same person, alone and guarded, complained wildly of a staring thing which bit and chewed at her.',
     'My life might have been passed in ease and luxury, but I preferred glory to every enticement that wealth placed in my path.',
     'I remember still more distinctly, that while he was pronounced by all parties at first sight "the most remarkable man in the world," no person made any attempt at accounting for his opinion.',
     'Men have felt the tears of the gods on white capped Thurai, though they have thought it rain; and have heard the sighs of the gods in the plaintive dawn winds of Lerion.',
     'Some reflection seemed to sting him, and the spasm of pain that for a moment convulsed his countenance, checked my indignation.',
     'No trees of any magnitude are to be seen.',
     'The sweep of the pendulum had increased in extent by nearly a yard.',
     'Besides, the body would not be even approximately fresh the next night.',
     'The chamber was full but there was no Protector; and there was an austere discontent manifest on the countenances of the leaders, and a whispering and busy tattle among the underlings, not less ominous.',
     'After marriage, however, this gentleman neglected, and, perhaps, even more positively ill treated her.',
     'Gilman mechanically attended classes that morning, but was wholly unable to fix his mind on his studies.',
     'I went so far as to say that I felt assured of her love; while I offered this assurance, and my own intensity of devotion, as two excuses for my otherwise unpardonable conduct.',
     'I had just consummated an unusually hearty dinner, of which the dyspeptic truffe formed not the least important item, and was sitting alone in the dining room, with my feet upon the fender, and at my elbow a small table which I had rolled up to the fire, and upon which were some apologies for dessert, with some miscellaneous bottles of wine, spirit and liqueur.',
     'One Survivor and Dead Man Found Aboard.',
     'Not from each other did Kalos and Musides conceal their work, but the sight was for them alone.',
     'Justine assumed an air of cheerfulness, while she with difficulty repressed her bitter tears.',
     'They were sensations, yet within them lay unbelievable elements of time and space things which at bottom possess no distinct and definite existence.',
     'When Carter left, he had said he was going to visit his old ancestral country around Arkham.',
     'The landlady met her in the passage; the poor creature asked, "Is my husband here?',
     'But now his oversensitive ears caught something behind him, and he looked back across the level terrace.',
     'By and by these are taught to carry parcels of some weight and this weight is gradually increased.',
     'Then hoary Nodens reached forth a wizened hand and helped Olney and his host into the vast shell, whereat the conches and the gongs set up a wild and awesome clamour.',
     'He said little, and that moodily, and with evident effort.',
     'A paralysis of fear stifled all attempts to cry out.',
     'I say that he will never be bodiless.',
     'False was all this false all but the affections of our nature, and the links of sympathy with pleasure or pain.',
     'In the most rugged of wildernesses in the most savage of the scenes of pure nature there is apparent the art of a creator; yet this art is apparent to reflection only; in no respect has it the obvious force of a feeling.',
     'I took him up at last, and threw him to about half a dozen yards from the balloon.',
     'And the question to be solved proceeds, or should proceed, to its final determination, by a succession of unerring steps liable to no change, and subject to no modification.',
     'What said he? some broken sentences I heard.',
     'The bandage lay heavily about the mouth but then might it not be the mouth of the breathing Lady of Tremaine?',
     'Then one night when the moon was full the travellers came to a mountain crest and looked down upon the myriad lights of Oonai.',
     'The sophists of the negative school, who, through inability to create, have scoffed at creation, are now found the loudest in applause.',
     'We expelled the bodies through the double hatches and were alone in the U .',
     "Oft when I have listened with gasping attention for the sound of the ocean mingled with my father's groans; and then wept untill my strength was gone and I was calm and faint, when I have recollected all this I have asked myself if this were not madness.",
     "C Fletcher's comedy of the Captain.",
     'I heard many things in hell.',
     '"Then there was \'The Involuntary Experimentalist,\' all about a gentleman who got baked in an oven, and came out alive and well, although certainly done to a turn.',
     'She paused, weeping, and then continued, "I thought with horror, my sweet lady, that you should believe your Justine, whom your blessed aunt had so highly honoured, and whom you loved, was a creature capable of a crime which none but the devil himself could have perpetrated.',
     'I shall do nothing rashly: you know me sufficiently to confide in my prudence and considerateness whenever the safety of others is committed to my care.',
     "Often with maternal affection she had figured their merits and talents exerted on life's wide stage.",
     '"Gentlemen," I said at last, as the party ascended the steps, "I delight to have allayed your suspicions.',
     "As the last echo ceased, I stepped into B 's and inquired for Talbot.",
     '"Why it did not seem altogether right to leave the interior blank that would have been insulting.',
     'Nothing in human shape could have destroyed the fair child.',
     'I alighted at Perth; and, though much fatigued by a constant exposure to the air for many hours, I would not rest, but merely altering my mode of conveyance, I went by land instead of air, to Dunkeld.',
     'When, to agree with him, I said I was distrustful of the Authenticity of Ossian\'s ms, Mr. Johnson said: "That, Sir, does not do your Understanding particular Credit; for what all the Town is sensible of, is no great Discovery for a Grub Street Critick to make.',
     'Safie nursed her with the most devoted affection, but the poor girl died, and the Arabian was left alone, unacquainted with the language of the country and utterly ignorant of the customs of the world.',
     'I looked towards its completion with a tremulous and eager hope, which I dared not trust myself to question but which was intermixed with obscure forebodings of evil that made my heart sicken in my bosom.',
     'The first manifestations, although marked, are unequivocal.',
     "Back Bay isn't Boston it isn't anything yet, because it's had no time to pick up memories and attract local spirits.",
     'But I am detailing a chain of facts and wish not to leave even a possible link imperfect.',
     'I traversed the streets without any clear conception of where I was or what I was doing.',
     'In the present instance, had the gold been gone, the fact of its delivery three days before would have formed something more than a coincidence.',
     '"C\'est à vous à faire," said his Majesty, cutting.',
     'I was weary with watching and for some time I had combated with the heavy sleep that weighed down my eyelids: but now, no longer fearful, I threw myself on my bed.',
     'Afterwards, succession runs thus: a o i d h n r s t u y c f g l m w b k p q x z.',
     'But in this existence, I dreamed that I should be at once cognizant of all things, and thus at once be happy in being cognizant of all.',
     'The inhabitants of our side of the moon have, evidently, no darkness at all, so there can be nothing of the "extremes" mentioned.',
     'I took a circuitous path, principally for the sake of going to the top of the mount before mentioned, which commanded a view of the city.',
     'When once disease was introduced into the rural districts, its effects appeared more horrible, more exigent, and more difficult to cure, than in towns.',
     'Especially was he afraid to be out of doors alone when the stars were shining, and if forced to this condition he would often glance furtively at the sky as if hunted by some monstrous thing therein.',
     '"This day was passed in the same routine as that which preceded it.',
     'As the afternoon advanced, it became increasingly difficult to see; and we heard the rumble of a thunderstorm gathering over Tempest Mountain.',
     'In the manner of my friend I was at once struck with an incoherence an inconsistency; and I soon found this to arise from a series of feeble and futile struggles to overcome an habitual trepidancy an excessive nervous agitation.',
     'In this manner many appalling hours passed; several of my dogs died, and I myself was about to sink under the accumulation of distress when I saw your vessel riding at anchor and holding forth to me hopes of succour and life.',
     'My preference would be to avoid Paine, since the fire station there might be open all night.',
     'Chords, vibrations, and harmonic ecstasies echoed passionately on every hand; while on my ravished sight burst the stupendous spectacle of ultimate beauty.',
     'And the roof is of pure gold, set upon tall pillars of ruby and azure, and having such carven figures of gods and heroes that he who looks up to those heights seems to gaze upon the living Olympus.',
     'No one ventured on board the vessel, and strange sights were averred to be seen at night, walking the deck, and hanging on the masts and shrouds.',
     'I must not forget one incident that occurred during this visit to London.',
     'We have forgotten what we did when she was not.',
     'Not a speck on their surface not a shade on their enamel not an indenture in their edges but what that period of her smile had sufficed to brand in upon my memory.',
     'The place was avoided with doubled assiduousness, and invested with every whispered myth tradition could supply.',
     'Above all was the sense of hearing acute.',
     "It seems that on that last hideous night Joe had stooped to look at the crimson rat tracks which led from Gilman's couch to the nearby hole.",
     "We may be instructed to build an Odyssey, but it is in vain that we are told how to conceive a 'Tempest,' an 'Inferno,' a 'Prometheus Bound,' a 'Nightingale,' such as that of Keats, or the 'Sensitive Plant' of .",
     "'Oppodeldoc,' whoever he is, is entirely devoid of imagination and imagination, in our humble opinion, is not only the soul of sy, but also its very heart.",
     'He will tell me " "Luchesi cannot tell Amontillado from Sherry." "And yet some fools will have it that his taste is a match for your own." "Come, let us go." "Whither?"',
     'I collected bones from charnel houses and disturbed, with profane fingers, the tremendous secrets of the human frame.',
     'This burst of passionate feeling over, with calmed thoughts we sat together, talking of the past and present.',
     'Dropping of its own accord upon his exit or perhaps purposely closed, it had become fastened by the spring; and it was the retention of this spring which had been mistaken by the police for that of the nail, farther inquiry being thus considered unnecessary.',
     '"Well, if you must have it so I will take a small reward just to satisfy your scruples.',
     'I did; but the fragile spirit clung to its tenement of clay for many days, for many weeks and irksome months, until my tortured nerves obtained the mastery over my mind, and I grew furious through delay, and, with the heart of a fiend, cursed the days and the hours and the bitter moments, which seemed to lengthen and lengthen as her gentle life declined, like shadows in the dying of the day.',
     'Here it is at first difficult to perceive the intention of the reasoner.',
     'To carry out his views, he solicited and obtained the patronage of Sir Everard Bringhurst and Mr. Osborne, two gentlemen well known for scientific acquirement, and especially for the interest they have exhibited in the progress of ærostation.',
     'For without warning, in one of the small hours beyond midnight, all the ravages of the years and the storms and the worms came to a tremendous climax; and after the crash there was nothing left standing in The Street save two ancient chimneys and part of a stout brick wall.',
     'I have little to record, except the fact to me quite a surprising one that, at an elevation equal to that of Cotopaxi, I experienced neither very intense cold, nor headache, nor difficulty of breathing; neither, I find, did Mr. Mason, nor Mr. Holland, nor Sir Everard.',
     'It was nearly dead calm when the voyagers first came in view of the coast, which was immediately recognized by both the seamen, and by Mr. Osborne.',
     'My steps were sure, and could afford but a single result.',
     'What was it I paused to think what was it that so unnerved me in the contemplation of the House of Usher?',
     'The whole house, with its wings, was constructed of the old fashioned Dutch shingles broad, and with unrounded corners.',
     'A change fell upon all things.',
     'And then, like David, I would try music to win the evil spirit from him; and once while singing I lifted my eyes towards him and saw his fixed on me and filled with tears; all his muscles seemed relaxed to softness.',
     'It was evident that my considerate friend, il fanatico, had quite forgotten his appointment with myself had forgotten it as soon as it was made.',
     'Parliament was divided by three factions, aristocrats, democrats, and royalists.',
     'But if the sentiment on which the fabric of her existence was founded, became common place through participation, the endless succession of attentions and graceful action snapt by transfer, his universe of love wrested from her, happiness must depart, and then be exchanged for its opposite.',
     'It was, the Belgian averred, a most extraordinary object; an object quite beyond the power of a layman to classify.',
     'I did not like the way he looked at healthy living bodies; and then there came a nightmarish session in the cellar laboratory when I learned that a certain specimen had been a living body when he secured it.',
     'I said it at first and I say it still, and I never swerved an inch, either, when he shewed that "Ghoul Feeding".',
     'Their minister at Constantinople was urged to make the necessary perquisitions, and should his existence be ascertained, to demand his release.',
     'I therefore undid only a few of these loops at one time, leaving the car suspended by the remainder.',
     'Whilst they strove to strip from life its embroidered robes of myth, and to shew in naked ugliness the foul thing that is reality, Kuranes sought for beauty alone.',
     'Unwholesome recollections of things in the Necronomicon and the Black Book welled up, and he found himself swaying to infandous rhythms said to pertain to the blackest ceremonies of the Sabbat and to have an origin outside the time and space we comprehend.',
     'Idris was too much taken up by her own dreadful fears, to be angry, hardly grieved; for she judged that insensibility must be the source of this continued rancour.',
     'The boundaries which divide Life from Death are at best shadowy and vague.',
     'All denied a part in the ritual murders, and averred that the killing had been done by Black Winged Ones which had come to them from their immemorial meeting place in the haunted wood.',
     'No woman could have inflicted the blows with any weapon.',
     'They were that kind the old lattice windows that went out of use before .',
     'Their happiness was not decreased by the absence of summer.',
     'The light curling waves bore us onward, and old ocean smiled at the freight of love and hope committed to his charge; it stroked gently its tempestuous plains, and the path was smoothed for us.',
     'The Countess had failed in this design with regard to her children; perhaps she hoped to find the next remove in birth more tractable.',
     'The bones of the tiny paws, it is rumoured, imply prehensile characteristics more typical of a diminutive monkey than of a rat; while the small skull with its savage yellow fangs is of the utmost anomalousness, appearing from certain angles like a miniature, monstrously degraded parody of a human skull.',
     'His intellect found sufficient field for exercise in his domestic circle, whose members, all adorned by refinement and literature, were many of them, like himself, distinguished by genius.',
     'It was, indeed, a tempestuous yet sternly beautiful night, and one wildly singular in its terror and its beauty.',
     'He often left us, and wandered by himself in the woods, or sailed in his little skiff, his books his only companions.',
     "Morbid art doesn't shock me, and when a man has the genius Pickman had I feel it an honour to know him, no matter what direction his work takes.",
     'Return; dearest one, you promised me this boon, that I should bring you health.',
     'I then took opportunities of conveying by night, to a retired situation east of Rotterdam, five iron bound casks, to contain about fifty gallons each, and one of a larger size; six tinned ware tubes, three inches in diameter, properly shaped, and ten feet in length; a quantity of a particular metallic substance, or semi metal, which I shall not name, and a dozen demijohns of a very common acid.',
     'Down this new opening the eye cannot penetrate very far; for the stream, accompanied by the wall, still bends to the left, until both are swallowed up by the leaves.',
     'Ye gods and what do I behold is that the departed spirit, the shade, the ghost, of my beloved puppy, which I perceive sitting with a grace so melancholy, in the corner?',
     'From those blurred and fragmentary memories we may infer much, yet prove little.',
     "Then again she sadly lamented her hard fate; that a woman, with a woman's heart and sensibility, should be driven by hopeless love and vacant hopes to take up the trade of arms, and suffer beyond the endurance of man privation, labour, and pain the while her dry, hot hand pressed mine, and her brow and lips burned with consuming fire.",
     'She had gone to hide her weakness; escaping from the castle, she had descended to the little park, and sought solitude, that she might there indulge her tears; I found her clinging round an old oak, pressing its rough trunk with her roseate lips, as her tears fell plenteously, and her sobs and broken exclamations could not be suppressed; with surpassing grief I beheld this loved one of my heart thus lost in sorrow I drew her towards me; and, as she felt my kisses on her eyelids, as she felt my arms press her, she revived to the knowledge of what remained to her. "You are very kind not to reproach me," she said: "I weep, and a bitter pang of intolerable sorrow tears my heart.',
     'Perdita looked at him like one amazed; her expressive countenance shone for a moment with tenderness; to see him only was happiness.',
     'While he spoke, so profound was the stillness that one might have heard a pin drop upon the floor.',
     'Of my exact age, even down to days and hours, I kept a most careful record, for each movement of the pendulum of the massive clock in the library told off so much more of my doomed existence.',
     'They were loud and quick unequal spoken apparently in fear as well as in anger.',
     'What he said was unintelligible, but words were uttered; the syllabification was distinct.',
     '"But when I saw you become the object of another\'s love; when I imagined that you might be loved otherwise than as a sacred type and image of loveliness and excellence; or that you might love another with a more ardent affection than that which you bore to me, then the fiend awoke within me; I dismissed your lover; and from that moment I have known no peace.',
     'Shakspeare, whose popularity was established by the approval of four centuries, had not lost his influence even at this dread period; but was still "Ut magus," the wizard to rule our hearts and govern our imaginations.',
     'For about half an hour the conversation ran upon ordinary topics, but at last, we contrived, quite naturally, to give it the following turn: CAPT.',
     'In August, the plague had appeared in the country of England, and during September it made its ravages.',
     "The writer professes to have translated his work from the English of one Mr. D'Avisson Davidson?",
     'He became an adventurer in the Greek wars.',
     'But, if the contest have proceeded thus far, it is the shadow which prevails, we struggle in vain.',
     'Had the body been in any respect despoiled?',
     'It had already buried its sharp edge a full inch in my flesh, and my sensations grew indistinct and confused.',
     'No fancy may picture the sublimity which might have been exhibited by a similar phenomenon taking place amid the darkness of the night.',
     "Looking around me during a pause in the Baron's discourse of which my readers may gather some faint idea when I say that it bore resemblance to the fervid, chanting, monotonous, yet musical sermonic manner of Coleridge, I perceived symptoms of even more than the general interest in the countenance of one of the party.",
     '"Wherefore do I feel thus?',
     'I replied to the yells of him who clamored.',
     'At first it told to me only the plain little tales of calm beaches and near ports, but with the years it grew more friendly and spoke of other things; of things more strange and more distant in space and in time.',
     '"My days were spent in close attention, that I might more speedily master the language; and I may boast that I improved more rapidly than the Arabian, who understood very little and conversed in broken accents, whilst I comprehended and could imitate almost every word that was spoken.',
     '"Here, then, we leave, in the very beginning, the groundwork for something more than a mere guess.',
     'Now, he reflected, those nervous fears were being mirrored in his disordered dreams.',
     'But perfect happiness is an attribute of angels; and those who possess it, appear angelic.',
     'It was a good overcoat.',
     'I adverted to what Adrian had already done I promised the same vigilance in furthering all his views.',
     "It was demonstrated, that the density of the comet's nucleus was far less than that of our rarest gas; and the harmless passage of a similar visitor among the satellites of Jupiter was a point strongly insisted upon, and which served greatly to allay terror.",
     'The last beams of the nearly sunken sun shot up from behind the far summit of Mount Athos; the sea of Marmora still glittered beneath its rays, while the Asiatic coast beyond was half hid in a haze of low cloud.',
     '"True," I observed; "the paper is clearly then upon the premises.',
     '"Twas dis eye, massa de lef eye jis as you tell me," and here it was his right eye that the negro indicated.',
     'As long as their disunion remained a secret, he cherished an expectation of re awakening past tenderness in her bosom; now that we were all made acquainted with these occurrences, and that Perdita, by declaring her resolves to others, in a manner pledged herself to their accomplishment, he gave up the idea of re union as futile, and sought only, since he was unable to influence her to change, to reconcile himself to the present state of things.',
     'Blew it, and tried again no go.',
     '"How died he?" "In his rash exertions to rescue a favorite portion of his hunting stud, he has himself perished miserably in the flames."',
     "'The things had all evidently been there,' he says, 'at least, three or four weeks, and there can be no doubt that the spot of this appalling outrage has been discovered.'",
     'The diddler approaches the bar of a tavern, and demands a couple of twists of tobacco.',
     "Open up the gates to Yog Sothoth with the long chant that ye'll find on page of the complete edition, an' then put a match to the prison.",
     "He said, I push'd every Aspirant off the Slopes of Parnassus.",
     'Has lived in Paris two years.',
     'I did so, but to little purpose, not being able to gather the least particle of meaning.',
     "We went up stairs into the chamber where the body of Mademoiselle L'Espanaye had been found, and where both the deceased still lay.",
     "A vulgar man that sometimes but he's deep.",
     'The squire who succeeded to it in studied sartain arts and made sartain discoveries, all connected with influences residing in this particular plot of ground, and eminently desarving of the strongest guarding.',
     'During this period, I became aware, for the first time, of the origin of the sulphurous light which illumined the cell.',
     'In relation to the second it is only necessary to repeat what we have before stated, that the machine is rolled about on castors, and will, at the request of a spectator, be moved to and fro to any portion of the room, even during the progress of a game.',
     'My host now took my hand to draw me to one of the two windows on the long side of the malodorous room, and at the first touch of his ungloved fingers I turned cold.',
     'I do not know how I came to live on such a street, but I was not myself when I moved there.',
     'This idea pursued me and tormented me at every moment from which I might otherwise have snatched repose and peace.',
     'Wonderful likewise were the gardens made by Zokkar the olden king.',
     'No youthful congregation of gallant hearted boys thronged the portal of the college; sad silence pervaded the busy school room and noisy playground.',
     '"I will soon explain to what these feelings tended, but allow me now to return to the cottagers, whose story excited in me such various feelings of indignation, delight, and wonder, but which all terminated in additional love and reverence for my protectors for so I loved, in an innocent, half painful self deceit, to call them."',
     'But I had no bodily, no visible, audible, or palpable presence.',
     'Yog Sothoth is the key and guardian of the gate.',
     'When exceptions did occur, they were mostly persons with no trace of aberrancy, like the old clerk at the hotel.',
     'How mutable are our feelings, and how strange is that clinging love we have of life even in the excess of misery I constructed another sail with a part of my dress and eagerly steered my course towards the land.',
     'Fallen houses choked up the streets.',
     'I was in the service of a farmer; and with crook in hand, my dog at my side, I shepherded a numerous flock on the near uplands.',
     'After years he began to call the slow sailing stars by name, and to follow them in fancy when they glided regretfully out of sight; till at length his vision opened to many secret vistas whose existence no common eye suspects.',
     'Her energy of character induced her still to combat with the ills of life; even those attendant on hopeless love presented themselves, rather in the shape of an adversary to be overcome, than of a victor to whom she must submit.',
     'A line of Marston\'s "Malcontent" Death\'s a good fellow and keeps open house struck me at that moment as a palpable lie.',
     'We occasioned the greatest excitement on board all an excitement greatly relished by ourselves, and especially by our two men, who, now under the influence of a dram of Geneva, seemed resolved to give all scruple, or fear, to the wind.',
     'It was he who had given me all the information I had of Tillinghast after I was repulsed in rage.',
     '"It was night, and the rain fell; and falling, it was rain, but, having fallen, it was blood.',
     'When Gilman climbed up a ladder to the cobwebbed level loft above the rest of the attic he found vestiges of a bygone aperture tightly and heavily covered with ancient planking and secured by the stout wooden pegs common in colonial carpentry.',
     'The maniac became composed; his person rose higher; authority beamed from his countenance.',
     'The eyes roll unnaturally in the head, without any corresponding motions of the lids or brows.',
     'Many were the waterfalls in their courses, and many were the lilied lakelets into which they expanded.',
     'Pale as marble, clear and beaming as that, she heard my tale, and enquired concerning the spot where he had been deposited.',
     '"Looking now, narrowly, through the cipher for combinations of known characters, we find, not very far from the beginning, this arrangement, , or egree, which, plainly, is the conclusion of the word \'degree,\' and gives us another letter, d, represented by .',
     'Nor could I ever after see the world as I had known it.',
     'All other matters and all different interests became absorbed in their single contemplation.',
     'While I was awake I knew what you meant by "spirit," but now it seems only a word such for instance as truth, beauty a quality, I mean.',
     'In the deeper dreams everything was likewise more distinct, and Gilman felt that the twilight abysses around him were those of the fourth dimension.',
     'The horse itself, in the foreground of the design, stood motionless and statue like while farther back, its discomfited rider perished by the dagger of a Metzengerstein.',
     'When they had retired to rest, if there was any moon or the night was star light, I went into the woods and collected my own food and fuel for the cottage.',
     'This did moderately well for a time; in fact, I was not avaricious, but my dog was.',
     'This was all true; but it was not less agonizing to take the admonition home.',
     'They talked of their illustrious Tyrant, and of the splendour of his capital; and exulted in the glory of the statue which Musides had wrought for him.',
     'Cast off the only gift that I have bestowed upon you, your grief, and rise from under my blighting influence as no flower so sweet ever did rise from beneath so much evil.',
     'But he was great once my fathair in Barcelona have hear of heem and only joost now he feex a arm of the plumber that get hurt of sudden.',
     'The effect produced by the firing of a cannon is that of simple vibration.',
     'Alas I knew not the desart it was about to reach; the rocks that would tear its waters, and the hideous scene that would be reflected in a more distorted manner in its waves.',
     'My heart beat quick as I approached the palings; my hand was on one of them, a leap would take me to the other side, when two keepers sprang from an ambush upon me: one knocked me down, and proceeded to inflict a severe horse whipping.',
     "The natives suspend it by a cord from the ceiling, and enjoy its fragrance for years.'",
     'My evil passions will have fled, for I shall meet with sympathy My life will flow quietly away, and in my dying moments I shall not curse my maker."',
     'A gentle breeze, however, now arose, as the sun was about descending; and while I remained standing on the brow of the slope, the fog gradually became dissipated into wreaths, and so floated over the scene.',
     'Soon after the departure of the couple, a gang of miscreants made their appearance, behaved boisterously, ate and drank without making payment, followed in the route of the young man and girl, returned to the inn about dusk, and re crossed the river as if in great haste.',
     'The position of the candelabrum displeased me, and outreaching my hand with difficulty, rather than disturb my slumbering valet, I placed it so as to throw its rays more fully upon the book.',
     '"Elizabeth Lavenza "Geneva, May th, " This letter revived in my memory what I had before forgotten, the threat of the fiend "I WILL BE WITH YOU ON YOUR WEDDING NIGHT" Such was my sentence, and on that night would the daemon employ every art to destroy me and tear me from the glimpse of happiness which promised partly to console my sufferings.',
     'He might have spoken, but I did not hear; one hand was stretched out, seemingly to detain me, but I escaped and rushed downstairs.',
     'Her face was exceedingly round, red, and full; and the same peculiarity, or rather want of peculiarity, attached itself to her countenance, which I before mentioned in the case of the president that is to say, only one feature of her face was sufficiently distinguished to need a separate characterization: indeed the acute Tarpaulin immediately observed that the same remark might have applied to each individual person of the party; every one of whom seemed to possess a monopoly of some particular portion of physiognomy.',
     'And the bright eyes of Eleonora grew brighter at my words; and she sighed as if a deadly burthen had been taken from her breast; and she trembled and very bitterly wept; but she made acceptance of the vow, for what was she but a child?',
     'I turned author myself.',
     'Bennett was asleep, having apparently felt the same anomalous drowsiness which affected me, so I designated Tobey for the next watch although even he was nodding.',
     'I contrived, however, to pacify them by promises of payment of all scores in full, as soon as I could bring the present business to a termination.',
     'Some who knew him do not admit that he ever existed.',
     'It was closer to Maple Hill than to Cone Mountain, some of the crude abodes indeed being dugouts on the side of the former eminence.',
     'The external world could take care of itself.',
     'The manipulations of Pompey had made, I must confess, a very striking difference in the appearance of the personal man.',
     'Then for a moment did Iranon believe he had found those who thought and felt even as he, though the town was not an hundredth as fair as Aira.',
     'Then, in endeavouring to do violence to my own disposition, I made all worse than before.',
     'His graceful elocution enchained the senses of his hearers.',
     'The small, weather worn telephone poles carried only two wires.',
     'Posterity is no more; fame, and ambition, and love, are words void of meaning; even as the cattle that grazes in the field, do thou, O deserted one, lie down at evening tide, unknowing of the past, careless of the future, for from such fond ignorance alone canst thou hope for ease Joy paints with its own colours every act and thought.',
     'Mr. B. merely cuts out and intersperses.',
     'But, in fact, this is a point of minor importance.',
     'One disgusting canvas seemed to depict a vast cross section of Beacon Hill, with ant like armies of the mephitic monsters squeezing themselves through burrows that honeycombed the ground.',
     'The theatre was tolerably well filled.',
     'I neither knew nor cared whether my experience was insanity, dreaming, or magic; but was determined to gaze on brilliance and gaiety at any cost.',
     'They are familiar to the world.',
     'She was nearly fifteen years older than he, and was the offspring of a former marriage of his father.',
     'I grew frantically mad, and struggled to force myself upward against the sweep of the fearful scimitar.',
     "M. St. Eustache, the lover and intended husband of Marie, who boarded in her mother's house, deposes that he did not hear of the discovery of the body of his intended until the next morning, when M. Beauvais came into his chamber and told him of it.",
     'I stopped automatically, though lacking the brain to retreat.',
     'When his gambols were over, I looked at the paper, and, to speak the truth, found myself not a little puzzled at what my friend had depicted.',
     'While such discussions were going on, their subject gradually approached, growing larger in apparent diameter, and of a more brilliant lustre.',
     'They stole off at first by ones and twos, then in larger companies, until, unimpeded by the officers, whole battalions sought the road that led to Macedonia.',
     'The scenery, judged by any ordinary aesthetic canon, is more than commonly beautiful; yet there is no influx of artists or summer tourists.',
     'I am glad that I have loved, and have experienced sympathetic joy and sorrow with my fellow creatures.',
     'The trees were lithe, mirthful, erect bright, slender, and graceful, of eastern figure and foliage, with bark smooth, glossy, and parti colored.',
     'Scarcely less savage was the "Lollipop," which thus discoursed: "Some individual, who rejoices in the appellation \'Oppodeldoc,\' to what low uses are the names of the illustrious dead too often applied has enclosed us some fifty or sixty verses commencing after this fashion: Achilles\' wrath, to Greece the direful spring Of woes unnumbered, c., c., c., c. "\'Oppodeldoc,\' whoever he is, is respectfully informed that there is not a printer\'s devil in our office who is not in the daily habit of composing better lines.',
     'He was wholly alone, and his first act was to walk to the balustrade and look dizzily down at the endless, Cyclopean city almost two thousand feet below.',
     'I went up to her and offered my services.',
     'Those who had lacked something lacked it no longer, yet did fear and hatred and ignorance still brood over The Street; for many had stayed behind, and many strangers had come from distant places to the ancient houses.',
     'This I perceived with unhealthy sharpness despite the fact that two of my other senses were violently assailed.',
     'He admitted that the principles that I laid down were the best; but he denied that they were the only ones.',
     'They were fearfully they were inconceivably hideous; but out of Evil proceeded Good; for their very excess wrought in my spirit an inevitable revulsion.',
     'And when time shall have softened your despair, new and dear objects of care will be born to replace those of whom we have been so cruelly deprived."',
     'How could I have suspected the thing I was to behold?',
     'One of these fragile mirrors, that ever doted on thine image, is about to be broken, crumbled to dust.',
     'I recall that I did not regard it as a common flashlight indeed, I had a common flashlight in another pocket.',
     '"One of the most popular pieces of mechanism which we have seen, Is the Magician constructed by M. Maillardet, for the purpose of answering certain given questions.',
     'He was a strange, furtive creature who constantly looked over his shoulder as if afraid of something, and when sober could not be persuaded to talk at all with strangers.',
     'Calderon de la Barca.',
     'I did not expect it, either, for I thought I was thoroughly forewarned regarding what the jewellery would turn out to be.',
     'He was detained three days longer and then he hastened to her.',
     'And why should I describe a sorrow which all have felt, and must feel?',
     'But I paused when I reflected on the story that I had to tell.',
     'Ruined castles hanging on the precipices of piny mountains, the impetuous Arve, and cottages every here and there peeping forth from among the trees formed a scene of singular beauty.',
     'I did not weep, but I wiped the perspiration from my brow, and tried to still my brain and heart beating almost to madness.',
     'The steps were many, and led to a narrow stone flagged passage which I knew must be far underground.',
     'I had already decided not to abandon the quest for the lurking fear, for in my rash ignorance it seemed to me that uncertainty was worse than enlightenment, however terrible the latter might prove to be.',
     'It had lost, in a great measure, the deep tint of blue it had hitherto worn, being now of a grayish white, and of a lustre dazzling to the eye.',
     'But now I went to it in cold blood, and my heart often sickened at the work of my hands.',
     'The door marked I, it will be remembered, is still open.',
     'We hired a chaise here, and with four horses drove with speed through the storm.',
     "Remember we're dealing with a hideous world in which we are practically helpless. . . .",
     'Of the various tales that of aged Soames, the family butler, is most ample and coherent.',
     'It was at Rome, during the Carnival of , that I attended a masquerade in the palazzo of the Neapolitan Duke Di Broglio.',
     'Thus, we talked of them, and moralized, as with diminished numbers we returned to Windsor Castle.',
     'And it was the mournful influence of the unperceived shadow that caused him to feel although he neither saw nor heard to feel the presence of my head within the room.',
     'A bright light then pervades the cupboard, and the body of the man would be discovered if it were there.',
     'Through Asia, from the banks of the Nile to the shores of the Caspian, from the Hellespont even to the sea of Oman, a sudden panic was driven.',
     'It was not a wholesome landscape after dark, and I believe I would have noticed its morbidity even had I been ignorant of the terror that stalked there.',
     'I always knew you were no scientist Trembling, eh? Trembling with anxiety to see the ultimate things I have discovered?',
     'Should I yield to your entreaties and, I may add, to the pleadings of my own bosom would I not be entitled to demand of you a very a very little boon in return?"',
     'I did not wholly disagree with him theoretically, yet held vague instinctive remnants of the primitive faith of my forefathers; so that I could not help eyeing the corpse with a certain amount of awe and terrible expectation.',
     'Idris, the most affectionate wife, sister and friend, was a tender and loving mother.',
     'Then all motion, of whatever nature, creates?',
     'The next night, without daring to ask for the rudder, he removes it.',
     'But, my dear Frankenstein," continued he, stopping short and gazing full in my face, "I did not before remark how very ill you appear; so thin and pale; you look as if you had been watching for several nights."',
     'He shrieked once once only.',
     'She first assured him of her boundless confidence; of this he must be conscious, since but for that she would not seek to detain him.',
     'Wild visions, opium engendered, flitted, shadow like, before me.',
     'Upon entering, I thrust him furiously from me.',
     'The tread was heavy, yet seemed to contain a curious quality of cautiousness; a quality which I disliked the more because the tread was heavy.',
     'Besides, the estates, which were contiguous, had long exercised a rival influence in the affairs of a busy government.',
     'She had been in his employ about a year, when her admirers were thrown info confusion by her sudden disappearance from the shop.',
     'He had apparently been strangled, for there was no sign of any violence except the black mark of fingers on his neck.',
     'Was that a rat I saw skulking into his hole?',
     '"Abhorred monster Fiend that thou art The tortures of hell are too mild a vengeance for thy crimes.',
     'Razor in hand, and fully lathered, it was sitting before a looking glass, attempting the operation of shaving, in which it had no doubt previously watched its master through the key hole of the closet.',
     '"We will resume this question by mere allusion to the revolting details of the surgeon examined at the inquest.',
     'Ah let me see Let me remember Yes; full easily do I call to mind the precise words of the dear promise you made to Eugenie last night.',
     'They partook less of transport and more of calm enthusiasm of enthusiastic repose.',
     'And now, too, hearing an incredible popping and fizzing of champagne, I discovered at length, that it proceeded from the person who performed the bottle of that delicate drink during dinner.',
     "The topography throughout, even when professing to accord with Blunt's Lunar Chart, is entirely at variance with that or any other lunar chart, and even grossly at variance with itself.",
     '"Vell, Monsieur," said she, after surveying me, in great apparent astonishment, for some moments "Vell, Monsieur?',
     'No, all must be changed.',
     'Agatha listened with respect, her eyes sometimes filled with tears, which she endeavoured to wipe away unperceived; but I generally found that her countenance and tone were more cheerful after having listened to the exhortations of her father.',
     'Madness rides the star wind . . .',
     'Nothing could be more magnificent.',
     'It was a very capital system indeed simple neat no trouble at all in fact it was delicious it was."',
     'Still she felt sure that he would come at last; and the wider the breach might appear at this crisis, the more secure she was of closing it for ever.',
     '"Well, then, Bobby, my boy you\'re a fine fellow, aren\'t you?',
     'This was the dream in which I saw a shoggoth for the first time, and the sight set me awake in a frenzy of screaming.',
     'The oven, for instance, that was a good hit.',
     'It was agreed to call the whole thing a chemical laboratory if discovery should occur.',
     'I left at once the employment of Messrs. Cut Comeagain, and set up in the Eye Sore line by myself one of the most lucrative, respectable, and independent of the ordinary occupations.',
     'I had imagined that the habitual endurance of the atmospheric pressure at the surface of the earth was the cause, or nearly so, of the pain attending animal existence at a distance above the surface.',
     'I was beyond measure disturbed by this intelligence.',
     'The patient, Mr. Edward Stapleton, had died, apparently of typhus fever, accompanied with some anomalous symptoms which had excited the curiosity of his medical attendants.',
     'Waterfront scum was far too common for special mention; though there was vague talk about one inland trip these mongrels had made, during which faint drumming and red flame were noted on the distant hills.',
     "On Frederick's lip arose a fiendish expression, as he became aware of the direction which his glance had, without his consciousness, assumed.",
     'This was, in fact, one of his hobbies.',
     'The mountain upon whose top we sit is Helseggen, the Cloudy.',
     'Matters of little moment are rarely consigned to parchment; since, for the mere ordinary purposes of drawing or writing, it is not nearly so well adapted as paper.',
     'Here he knew strange things had happened once, and there was a faint suggestion behind the surface that everything of that monstrous past might not at least in the darkest, narrowest, and most intricately crooked alleys have utterly perished.',
     'Nobody but my enemies ever calls me Suky Snobbs.',
     '"Your arrival, my dear cousin," said she, "fills me with hope.',
     'What, if circumstance should lead Perdita to suspect, and suspecting to be resolved?',
     'Of pain there was some little; of pleasure there was much; but of moral pain or pleasure none at all.',
     'I felt I had known it before, in a past remote beyond all recollection; beyond even my tenancy of the body I now possess.',
     'Three coffin heights, he reckoned, would permit him to reach the transom; but he could do better with four.',
     'By this time Dombrowski, Choynski, Desrochers, Mazurewicz, and the top floor lodger were all crowding into the doorway, and the landlord had sent his wife back to telephone for Dr. Malkowski.',
     'The wind, prince of air, raged through his kingdom, lashing the sea into fury, and subduing the rebel earth into some sort of obedience.',
     'The season of the assizes approached.',
     'None of the non natives ever stayed out late at night, there being a widespread impression that it was not wise to do so.',
     '"But is it not possible," I suggested, "that although the letter may be in possession of the minister, as it unquestionably is, he may have concealed it elsewhere than upon his own premises?"',
     'He had awaked to find himself standing bloody handed in the snow before his cabin, the mangled corpse of his neighbour Peter Slader at his feet.',
     'The jumble of French changed to a cry in English, and the hoarse voice shouted excitedly, "My breath, my breath" Then the awakening became complete, and with a subsidence of facial expression to the normal state my uncle seized my hand and began to relate a dream whose nucleus of significance I could only surmise with a kind of awe.',
     'And the queer part was, that Pickman got none of his power from the use of selectiveness or bizarrerie.',
     '\'By your language, stranger, I suppose you are my countryman; are you French?\' "\'No; but I was educated by a French family and understand that language only.',
     'We rapidly drew near, so that at length the number and forms of those within could be discerned; its dark sides grew big, and the splash of its oars became audible: I could distinguish the languid form of my friend, as he half raised himself at our approach.',
     'It was certainly nervous waiting, and the blasphemous book in my hands made it doubly so.',
     'Perdita was still to a great degree uneducated.',
     'It was the silent first comer who had burned his books.',
     '"To dream," he continued, resuming the tone of his desultory conversation, as he held up to the rich light of a censer one of the magnificent vases "to dream has been the business of my life.',
     'It was owned by a curious group of half castes whose frequent meetings and night trips to the woods attracted no little curiosity; and it had set sail in great haste just after the storm and earth tremors of March st.',
     'Is it any wonder, then, that I prize it?',
     'All this time I had never let go of the ring bolt.',
     'I feared to wander from the sight of my fellow creatures lest when alone he should come to claim his companion.',
     'I have said that the sole effect of my somewhat childish experiment that of looking down within the tarn had been to deepen the first singular impression.',
     'Olney does not recall many of the wonders he told, or even who he was; but says that he was strange and kindly, and filled with the magic of unfathomed voids of time and space.',
     'After all we miscalculated.',
     'His health was impaired beyond hope of cure; and it became his earnest wish, before he died, to preserve his daughter from the poverty which would be the portion of her orphan state.',
     'So now I am to end it all, having written a full account for the information or the contemptuous amusement of my fellow men.',
     'Young men should be diffident of themselves, you know, M. Clerval: I was myself when young; but that wears out in a very short time."',
     'I reasoned, for example, thus: When I drew the scarabæus, there was no skull apparent upon the parchment.',
     'Had I never quitted Windsor, these emotions would not have been so intense; but I had in Greece been the prey of fear and deplorable change; in Greece, after a period of anxiety and sorrow, I had seen depart two, whose very names were the symbol of greatness and virtue.',
     'Although no two of the time pieces in the chamber struck the individual seconds accurately together, yet I had no difficulty in holding steadily in mind the tones, and the respective momentary errors of each.',
     "That's capital That will do for the similes.",
     'Oh Not the ten thousandth portion of the anguish that was mine during the lingering detail of its execution.',
     'Now and then, beneath the brown pall of leaves that rotted and festered in the antediluvian forest darkness, I could trace the sinister outlines of some of those low mounds which characterised the lightning pierced region.',
     '"And do you dream?" said the daemon.',
     'I permitted it to do so; occasionally stooping and patting it as I proceeded.',
     'I lived in a temple glorified by intensest sense of devotion and rapture; I walked, a consecrated being, contemplating only your power, your excellence; For O, you stood beside me, like my youth, Transformed for me the real to a dream, Cloathing the palpable and familiar With golden exhalations of the dawn.',
     'Many went up as high as Belfast to ensure a shorter passage, and then journeying south through Scotland, they were joined by the poorer natives of that country, and all poured with one consent into England.',
     'Induction, a posteriori, would have brought phrenology to admit, as an innate and primitive principle of human action, a paradoxical something, which we may call perverseness, for want of a more characteristic term.',
     'But it made men dream, and so they knew enough to keep away.',
     'In truth, much as the owners of cats hated these odd folk, they feared them more; and instead of berating them as brutal assassins, merely took care that no cherished pet or mouser should stray toward the remote hovel under the dark trees.',
     'With the lady in question this portion proved to be the mouth.',
     'But it was not so; thou didst seek my extinction, that I might not cause greater wretchedness; and if yet, in some mode unknown to me, thou hadst not ceased to think and feel, thou wouldst not desire against me a vengeance greater than that which I feel.',
     'A bold diddle is this.',
     'With sentiments of profound respect, Your most obedient servant, VON JUNG.',
     'The obtuse instrument was clearly the stone pavement in the yard, upon which the victim had fallen from the window which looked in upon the bed.',
     'In the architecture and embellishments of the chamber, the evident design had been to dazzle and astound.',
     'Among these, and highly distinguished by her, was Prince Zaimi, ambassador to England from the free States of Greece; and his daughter, the young Princess Evadne, passed much of her time at Windsor Castle.',
     '"Another circumstance strengthened and confirmed these feelings.',
     "Secondly, having settled it to be God's will that man should continue his species, we discovered an organ of amativeness, forthwith.",
     'At first, I had spoken only to those nearest me; but the whole assembly gathered about me, and I found that I was listened to by all.',
     'Of polished desert quarried marble were its walls, in height cubits and in breadth , so that chariots might pass each other as men drave them along the top.',
     '"You have your fortune to make, Thingum," resumed Mr. Crab, "and that governor of yours is a millstone about your neck.',
     'His servants are by no means numerous.',
     'Shame seemed to hold him back; yet he evidently wished to establish a renewal of confidence and affection.',
     'I obtained from my father a respite of some weeks.',
     'We were in the most imminent peril, but as we could only remain passive, my chief attention was occupied by my unfortunate guest whose illness increased in such a degree that he was entirely confined to his bed.',
     'That object no larger than a good sized rat and quaintly called by the townspeople "Brown Jenkin" seemed to have been the fruit of a remarkable case of sympathetic herd delusion, for in no less than eleven persons had testified to glimpsing it.',
     'This may not be; cease to argue the point, for I cannot consent."',
     'He was perfectly self possessed; he accosted us both with courtesy, seemed immediately to enter into our feelings, and to make one with us.',
     'He again took my arm, and we proceeded.',
     'This slowly became merged in a sense of retirement this again in a consciousness of solitude.',
     'The box did not go into the extra state room.',
     'THE thousand injuries of Fortunato I had borne as I best could; but when he ventured upon insult, I vowed revenge.',
     'This second sight is very efficient when properly managed.',
     'For a moment my soul was elevated from its debasing and miserable fears to contemplate the divine ideas of liberty and self sacrifice of which these sights were the monuments and the remembrancers.',
     'I now began to experience, at intervals, severe pain in the head, especially about the ears still, however, breathing with tolerable freedom.',
     'Only the silent, sleepy, staring houses in the backwoods can tell all that has lain hidden since the early days; and they are not communicative, being loath to shake off the drowsiness which helps them forget.',
     'In such case I should have commenced with a collation and analysis of the shorter words, and, had a word of a single letter occurred, as is most likely, a or I, for example, I should have considered the solution as assured.',
     'Six years have passed since I resolved on my present undertaking.',
     'In a very short time he was out of sight, and I have no doubt he reached home in safety.',
     'The very boldness of his language gave him weight; each knew that he spoke truth a truth known, but not acknowledged.',
     'At Bates Street I drew into a yawning vestibule while two shambling figures crossed in front of me, but was soon on my way again and approaching the open space where Eliot Street obliquely crosses Washington at the intersection of South.',
     'We assembled again towards evening, and Perdita insisted on our having recourse to music.',
     'Then suddenly all the stars were blotted from the sky even bright Deneb and Vega ahead, and the lone Altair and Fomalhaut behind us.',
     '"I suppose you have called about the Ourang Outang.',
     'You well know that on the whole earth there is no sacrifise that I would not make, no labour that I would not undergo with the mere hope that I might bring you ease.',
     'It was formed of a single, broad and thick plank of the tulip wood.',
     'Here the bank slopes upward from the stream in a very gentle ascent, forming a broad sward of grass of a texture resembling nothing so much as velvet, and of a brilliancy of green which would bear comparison with the tint of the purest emerald.',
     'In the heart of one like me there are secret thoughts working, and secret tortures which you ought not to seek to discover.',
     'Supporting her as I did, still she lagged: and at the distance of half a mile, after many stoppages, shivering fits, and half faintings, she slipt from my supporting arm on the snow, and with a torrent of tears averred that she must be taken, for that she could not proceed.',
     'In the radical theory of reanimation they saw nothing but the immature vagaries of a youthful enthusiast whose slight form, yellow hair, spectacled blue eyes, and soft voice gave no hint of the supernormal almost diabolical power of the cold brain within.',
     'He measures two points, and, with a grace inimitable, offers his Majesty the choice.',
     'In especial, the slightest appearance of mystery of any point I cannot exactly comprehend puts me at once into a pitiable state of agitation.',
     'Like many excellent people, he seemed possessed with a spirit of tantalization, which might easily, at a casual glance, have been mistaken for malevolence.',
     'Upon the eighth night I was more than usually cautious in opening the door.',
     'Mrs. Frye proposed telephoning the neighbours, and Elmer was about to agree when the noise of splintering wood burst in upon their deliberations.',
     'I was faint, even fainter than the hateful modernity of that accursed city had made me.',
     'Their farther intentions were not ascertained; but we can safely promise our readers some additional information either on Monday or in the course of the next day, at farthest.',
     'In a few respects they are even unworthy of serious refutation.',
     'Yet how could I find this?',
     'Dr. Barnard, who had been watching the patient, thought he noticed in the pale blue eyes a certain gleam of peculiar quality; and in the flaccid lips an all but imperceptible tightening, as if of intelligent determination.',
     'Leaning upon the arm of the gallant Pompey, and attended at a respectable distance by Diana, I proceeded down one of the populous and very pleasant streets of the now deserted Edina.',
     'The writer spoke of acute bodily illness of a mental disorder which oppressed him and of an earnest desire to see me, as his best, and indeed his only personal friend, with a view of attempting, by the cheerfulness of my society, some alleviation of his malady.',
     'Immediately upon detecting this motion, and before the arm itself begins to move, let him withdraw his piece, as if perceiving an error in his manoeuvre.',
     'We kept track of all the deaths and their circumstances with systematic care.',
     'After much toil I found it.',
     'At length removing carefully his meerschaum from the right to the left corner of his mouth, he condescended to speak.',
     'But hoax, with these sort of people, is, I believe, a general term for all matters above their comprehension.',
     'She alone knew the weight which Raymond attached to his success.',
     'The town was now a city, and one by one the cabins gave place to houses; simple, beautiful houses of brick and wood, with stone steps and iron railings and fanlights over the doors.',
     'It was our plan to remain where we were and intercept the liner Dacia, mentioned in information from agents in New York.',
     'There were many palaces, the least of which were mightier than any in Thraa or Ilarnek or Kadatheron.',
     'It seems that he did not scorn the incantations of the mediaevalists, since he believed these cryptic formulae to contain rare psychological stimuli which might conceivably have singular effects on the substance of a nervous system from which organic pulsations had fled.',
     'She quitted her native Greece; her father died; by degrees she was cut off from all the companions and ties of her youth.',
     'He died in debt, and his little property was seized immediately by his creditors.',
     'I walked vigorously faster still faster at length I ran.',
     'We talked of the ravages made last year by pestilence in every quarter of the world; and of the dreadful consequences of a second visitation.',
     ...]




```python
#checking " quotes examples
[sent for sent in list(train['text']) if '"' in sent]
```


```python
#None of text contains author clases within sentences
for auth in list(set(train['author'])):
    print(auth)
    print([sent for sent in list(train['text']) if auth in sent])
```


```python
#checking most occuring words in sentences , most of word with highest frequency are stop words which will not help us in deciding the class
top = Counter([item for sublist in train['text'] for item in sublist.split(' ')])
print(top.most_common(20))
```


```python
#removing 
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    text = text.lower() #Lower - Casing
    text = unidecode.unidecode(text) #Removing accents e.g ô
    text = re.sub('[^A-Za-z]'," ",text) #removing special characters
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split(" ") if word!=""]) #lemmatizing and remocing extra spaces
    return text
```


```python
train['clean_text'] = train['text'].progress_apply(preprocess)
```


      0%|          | 0/19579 [00:00<?, ?it/s]



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
      <th>id</th>
      <th>text</th>
      <th>author</th>
      <th>len</th>
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id26305</td>
      <td>This process, however, afforded me no means of...</td>
      <td>EAP</td>
      <td>231</td>
      <td>this process however afforded me no mean of as...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id17569</td>
      <td>It never once occurred to me that the fumbling...</td>
      <td>HPL</td>
      <td>71</td>
      <td>it never once occurred to me that the fumbling...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id11008</td>
      <td>In his left hand was a gold snuff box, from wh...</td>
      <td>EAP</td>
      <td>200</td>
      <td>in his left hand wa a gold snuff box from whic...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id27763</td>
      <td>How lovely is spring As we looked from Windsor...</td>
      <td>MWS</td>
      <td>206</td>
      <td>how lovely is spring a we looked from windsor ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id12958</td>
      <td>Finding nothing else, not even gold, the Super...</td>
      <td>HPL</td>
      <td>174</td>
      <td>finding nothing else not even gold the superin...</td>
    </tr>
  </tbody>
</table>
</div>



# Feature extraction , Model building


```python
xtrain, xvalid, ytrain, yvalid = train_test_split(train, train['author'], 
                                                  stratify=train['author'], 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)
```


```python
label_encoder = LabelEncoder()
ytrain_encoded = label_encoder.fit_transform(ytrain)
yvalid_encoded = label_encoder.transform(yvalid)
```


```python
feats = ['clean_text','len']
```

# Using TFidf


```python
#this will also remove stop words
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(list(xtrain[feats[0]]) + list(xvalid[feats[0]]))
xtrain_tfv =  tfv.transform(xtrain[feats[0]]) 
xvalid_tfv = tfv.transform(xvalid[feats[0]])
```

    /opt/conda/lib/python3.10/site-packages/sklearn/utils/_param_validation.py:558: FutureWarning: Passing an int for a boolean parameter is deprecated in version 1.2 and won't be supported anymore in version 1.4.
      warnings.warn(



```python
tfv.get_feature_names_out()
```




    array(['abandon', 'abandoned', 'abandoning', ..., 'zit', 'zokkar', 'zone'],
          dtype=object)




```python
xtrain_tfv
```




    <15663x15524 sparse matrix of type '<class 'numpy.float64'>'
    	with 189775 stored elements in Compressed Sparse Row format>




```python
xtrain_tfv.todense()
```




    matrix([[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]])




```python
#Logistic regression
clf = LogisticRegression()
clf.fit(xtrain_tfv, ytrain_encoded)
```

    /opt/conda/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
#F1 score
f1_score(yvalid_encoded, clf.predict(xvalid_tfv), average='weighted')
```




    0.8032592565131291




```python

```

# Using GloVe 


```python
# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
```


    0it [00:00, ?it/s]


    Found 2196017 word vectors.



```python
embeddings_index['the'].shape
```




    (300,)




```python
embeddings_index['the']
```




    array([ 2.7204e-01, -6.2030e-02, -1.8840e-01,  2.3225e-02, -1.8158e-02,
            6.7192e-03, -1.3877e-01,  1.7708e-01,  1.7709e-01,  2.5882e+00,
           -3.5179e-01, -1.7312e-01,  4.3285e-01, -1.0708e-01,  1.5006e-01,
           -1.9982e-01, -1.9093e-01,  1.1871e+00, -1.6207e-01, -2.3538e-01,
            3.6640e-03, -1.9156e-01, -8.5662e-02,  3.9199e-02, -6.6449e-02,
           -4.2090e-02, -1.9122e-01,  1.1679e-02, -3.7138e-01,  2.1886e-01,
            1.1423e-03,  4.3190e-01, -1.4205e-01,  3.8059e-01,  3.0654e-01,
            2.0167e-02, -1.8316e-01, -6.5186e-03, -8.0549e-03, -1.2063e-01,
            2.7507e-02,  2.9839e-01, -2.2896e-01, -2.2882e-01,  1.4671e-01,
           -7.6301e-02, -1.2680e-01, -6.6651e-03, -5.2795e-02,  1.4258e-01,
            1.5610e-01,  5.5510e-02, -1.6149e-01,  9.6290e-02, -7.6533e-02,
           -4.9971e-02, -1.0195e-02, -4.7641e-02, -1.6679e-01, -2.3940e-01,
            5.0141e-03, -4.9175e-02,  1.3338e-02,  4.1923e-01, -1.0104e-01,
            1.5111e-02, -7.7706e-02, -1.3471e-01,  1.1900e-01,  1.0802e-01,
            2.1061e-01, -5.1904e-02,  1.8527e-01,  1.7856e-01,  4.1293e-02,
           -1.4385e-02, -8.2567e-02, -3.5483e-02, -7.6173e-02, -4.5367e-02,
            8.9281e-02,  3.3672e-01, -2.2099e-01, -6.7275e-03,  2.3983e-01,
           -2.3147e-01, -8.8592e-01,  9.1297e-02, -1.2123e-02,  1.3233e-02,
           -2.5799e-01, -2.9720e-02,  1.6754e-02,  1.3690e-02,  3.2377e-01,
            3.9546e-02,  4.2114e-02, -8.8243e-02,  3.0318e-01,  8.7747e-02,
            1.6346e-01, -4.0485e-01, -4.3845e-02, -4.0697e-02,  2.0936e-01,
           -7.7795e-01,  2.9970e-01,  2.3340e-01,  1.4891e-01, -3.9037e-01,
           -5.3086e-02,  6.2922e-02,  6.5663e-02, -1.3906e-01,  9.4193e-02,
            1.0344e-01, -2.7970e-01,  2.8905e-01, -3.2161e-01,  2.0687e-02,
            6.3254e-02, -2.3257e-01, -4.3520e-01, -1.7049e-02, -3.2744e-01,
           -4.7064e-02, -7.5149e-02, -1.8788e-01, -1.5017e-02,  2.9342e-02,
           -3.5270e-01, -4.4278e-02, -1.3507e-01, -1.1644e-01, -1.0430e-01,
            1.3920e-01,  3.9199e-03,  3.7603e-01,  6.7217e-02, -3.7992e-01,
           -1.1241e+00, -5.7357e-02, -1.6826e-01,  3.9410e-02,  2.6040e-01,
           -2.3866e-02,  1.7963e-01,  1.3553e-01,  2.1390e-01,  5.2633e-02,
           -2.5033e-01, -1.1307e-01,  2.2234e-01,  6.6597e-02, -1.1161e-01,
            6.2438e-02, -2.7972e-01,  1.9878e-01, -3.6262e-01, -1.0006e-05,
           -1.7262e-01,  2.9166e-01, -1.5723e-01,  5.4295e-02,  6.1010e-02,
           -3.9165e-01,  2.7660e-01,  5.7816e-02,  3.9709e-01,  2.5229e-02,
            2.4672e-01, -8.9050e-02,  1.5683e-01, -2.0960e-01, -2.2196e-01,
            5.2394e-02, -1.1360e-02,  5.0417e-02, -1.4023e-01, -4.2825e-02,
           -3.1931e-02, -2.1336e-01, -2.0402e-01, -2.3272e-01,  7.4490e-02,
            8.8202e-02, -1.1063e-01, -3.3526e-01, -1.4028e-02, -2.9429e-01,
           -8.6911e-02, -1.3210e-01, -4.3616e-01,  2.0513e-01,  7.9362e-03,
            4.8505e-01,  6.4237e-02,  1.4261e-01, -4.3711e-01,  1.2783e-01,
           -1.3111e-01,  2.4673e-01, -2.7496e-01,  1.5896e-01,  4.3314e-01,
            9.0286e-02,  2.4662e-01,  6.6463e-02, -2.0099e-01,  1.1010e-01,
            3.6440e-02,  1.7359e-01, -1.5689e-01, -8.6328e-02, -1.7316e-01,
            3.6975e-01, -4.0317e-01, -6.4814e-02, -3.4166e-02, -1.3773e-02,
            6.2854e-02, -1.7183e-01, -1.2366e-01, -3.4663e-02, -2.2793e-01,
           -2.3172e-01,  2.3900e-01,  2.7473e-01,  1.5332e-01,  1.0661e-01,
           -6.0982e-02, -2.4805e-02, -1.3478e-01,  1.7932e-01, -3.7374e-01,
           -2.8930e-02, -1.1142e-01, -8.3890e-02, -5.5932e-02,  6.8039e-02,
           -1.0783e-01,  1.4650e-01,  9.4617e-02, -8.4554e-02,  6.7429e-02,
           -3.2910e-01,  3.4082e-02, -1.6747e-01, -2.5997e-01, -2.2917e-01,
            2.0159e-02, -2.7580e-02,  1.6136e-01, -1.8538e-01,  3.7665e-02,
            5.7603e-01,  2.0684e-01,  2.7941e-01,  1.6477e-01, -1.8769e-02,
            1.2062e-01,  6.9648e-02,  5.9022e-02, -2.3154e-01,  2.4095e-01,
           -3.4710e-01,  4.8540e-02, -5.6502e-02,  4.1566e-01, -4.3194e-01,
            4.8230e-01, -5.1759e-02, -2.7285e-01, -2.5893e-01,  1.6555e-01,
           -1.8310e-01, -6.7340e-02,  4.2457e-01,  1.0346e-02,  1.4237e-01,
            2.5939e-01,  1.7123e-01, -1.3821e-01, -6.6846e-02,  1.5981e-02,
           -3.0193e-01,  4.3579e-02, -4.3102e-02,  3.5025e-01, -1.9681e-01,
           -4.2810e-01,  1.6899e-01,  2.2511e-01, -2.8557e-01, -1.0280e-01,
           -1.8168e-02,  1.1407e-01,  1.3015e-01, -1.8317e-01,  1.3230e-01])




```python
class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X,dtype=object)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
```


```python
class MeanEmbeddingVectorizer(object):
    def __init__(self, glove):
        self.glove = glove
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(glove['the'])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.glove[w] for w in words if w in self.glove]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
```


```python
mean_embedding_vectorizer = MeanEmbeddingVectorizer(embeddings_index)
mean_embedded = mean_embedding_vectorizer.fit_transform(xtrain[feats[0]])
```


```python
mean_embedded.shape
```




    (15663, 300)




```python
ytrain_encoded.shape
```




    (15663,)




```python
#Logistic regression
clf = LogisticRegression()
clf.fit(mean_embedded, ytrain_encoded)
```

    /opt/conda/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
f1_score(yvalid_encoded, clf.predict(mean_embedding_vectorizer.fit_transform(xvalid[feats[0]])), average='weighted')
```




    0.7273652493598247




```python

```

# Using LSTM


```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.models import load_model
```

    2024-02-28 05:45:14.408731: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-02-28 05:45:14.408858: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-02-28 05:45:14.568293: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered



```python
# Use the Keras tokenizer
num_words = 2000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(xtrain[feats[0]].values)
```


```python
# Pad the data 
X = tokenizer.texts_to_sequences(xtrain[feats[0]].values)
X = pad_sequences(X, maxlen=num_words)

X_valid = tokenizer.texts_to_sequences(xvalid[feats[0]].values)
X_valid = pad_sequences(X_valid, maxlen=num_words)
```


```python
X.shape
```




    (15663, 2000)




```python
# Build out our simple LSTM
embed_dim = 300
lstm_out = 196

# Model saving callback
ckpt_callback = ModelCheckpoint('keras_model', 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')
model = Sequential()
model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))
model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 2000, 300)         600000    
                                                                     
     lstm (LSTM)                 (None, 196)               389648    
                                                                     
     dense (Dense)               (None, 3)                 591       
                                                                     
    =================================================================
    Total params: 990239 (3.78 MB)
    Trainable params: 990239 (3.78 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    None



```python
ytrain_encoded_onehot = pd.get_dummies(ytrain_encoded).astype(int).values
yvalid_encoded_onehot = pd.get_dummies(yvalid_encoded).astype(int).values
```


```python
batch_size = 32
model.fit(X, ytrain_encoded_onehot, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])
```

    Epoch 1/8
    392/392 [==============================] - ETA: 0s - loss: 0.7499 - categorical_crossentropy: 0.7499
    Epoch 1: val_loss improved from inf to 0.57724, saving model to keras_model
    392/392 [==============================] - 2954s 8s/step - loss: 0.7499 - categorical_crossentropy: 0.7499 - val_loss: 0.5772 - val_categorical_crossentropy: 0.5772
    Epoch 2/8
    392/392 [==============================] - ETA: 0s - loss: 0.4861 - categorical_crossentropy: 0.4861
    Epoch 2: val_loss improved from 0.57724 to 0.57456, saving model to keras_model
    392/392 [==============================] - 2960s 8s/step - loss: 0.4861 - categorical_crossentropy: 0.4861 - val_loss: 0.5746 - val_categorical_crossentropy: 0.5746
    Epoch 3/8
    392/392 [==============================] - ETA: 0s - loss: 0.4140 - categorical_crossentropy: 0.4140
    Epoch 3: val_loss did not improve from 0.57456
    392/392 [==============================] - 2984s 8s/step - loss: 0.4140 - categorical_crossentropy: 0.4140 - val_loss: 0.6165 - val_categorical_crossentropy: 0.6165
    Epoch 4/8
    392/392 [==============================] - ETA: 0s - loss: 0.3476 - categorical_crossentropy: 0.3476
    Epoch 4: val_loss did not improve from 0.57456
    392/392 [==============================] - 3026s 8s/step - loss: 0.3476 - categorical_crossentropy: 0.3476 - val_loss: 0.6609 - val_categorical_crossentropy: 0.6609
    Epoch 5/8
    377/392 [===========================>..] - ETA: 1:49 - loss: 0.3015 - categorical_crossentropy: 0.3015


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[55], line 2
          1 batch_size = 32
    ----> 2 model.fit(X, ytrain_encoded_onehot, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])


    File /opt/conda/lib/python3.10/site-packages/keras/src/utils/traceback_utils.py:65, in filter_traceback.<locals>.error_handler(*args, **kwargs)
         63 filtered_tb = None
         64 try:
    ---> 65     return fn(*args, **kwargs)
         66 except Exception as e:
         67     filtered_tb = _process_traceback_frames(e.__traceback__)


    File /opt/conda/lib/python3.10/site-packages/keras/src/engine/training.py:1807, in Model.fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1799 with tf.profiler.experimental.Trace(
       1800     "train",
       1801     epoch_num=epoch,
       (...)
       1804     _r=1,
       1805 ):
       1806     callbacks.on_train_batch_begin(step)
    -> 1807     tmp_logs = self.train_function(iterator)
       1808     if data_handler.should_sync:
       1809         context.async_wait()


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/util/traceback_utils.py:150, in filter_traceback.<locals>.error_handler(*args, **kwargs)
        148 filtered_tb = None
        149 try:
    --> 150   return fn(*args, **kwargs)
        151 except Exception as e:
        152   filtered_tb = _process_traceback_frames(e.__traceback__)


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:832, in Function.__call__(self, *args, **kwds)
        829 compiler = "xla" if self._jit_compile else "nonXla"
        831 with OptionalXlaContext(self._jit_compile):
    --> 832   result = self._call(*args, **kwds)
        834 new_tracing_count = self.experimental_get_tracing_count()
        835 without_tracing = (tracing_count == new_tracing_count)


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:868, in Function._call(self, *args, **kwds)
        865   self._lock.release()
        866   # In this case we have created variables on the first call, so we run the
        867   # defunned version which is guaranteed to never create variables.
    --> 868   return tracing_compilation.call_function(
        869       args, kwds, self._no_variable_creation_config
        870   )
        871 elif self._variable_creation_config is not None:
        872   # Release the lock early so that multiple threads can perform the call
        873   # in parallel.
        874   self._lock.release()


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:139, in call_function(args, kwargs, tracing_options)
        137 bound_args = function.function_type.bind(*args, **kwargs)
        138 flat_inputs = function.function_type.unpack_inputs(bound_args)
    --> 139 return function._call_flat(  # pylint: disable=protected-access
        140     flat_inputs, captured_inputs=function.captured_inputs
        141 )


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py:1323, in ConcreteFunction._call_flat(self, tensor_inputs, captured_inputs)
       1319 possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
       1320 if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
       1321     and executing_eagerly):
       1322   # No tape is watching; skip to running the function.
    -> 1323   return self._inference_function.call_preflattened(args)
       1324 forward_backward = self._select_forward_and_backward_functions(
       1325     args,
       1326     possible_gradient_type,
       1327     executing_eagerly)
       1328 forward_function, args_with_tangents = forward_backward.forward()


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:216, in AtomicFunction.call_preflattened(self, args)
        214 def call_preflattened(self, args: Sequence[core.Tensor]) -> Any:
        215   """Calls with flattened tensor inputs and returns the structured output."""
    --> 216   flat_outputs = self.call_flat(*args)
        217   return self.function_type.pack_output(flat_outputs)


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:251, in AtomicFunction.call_flat(self, *args)
        249 with record.stop_recording():
        250   if self._bound_context.executing_eagerly():
    --> 251     outputs = self._bound_context.call_function(
        252         self.name,
        253         list(args),
        254         len(self.function_type.flat_outputs),
        255     )
        256   else:
        257     outputs = make_call_op_in_graph(
        258         self,
        259         list(args),
        260         self._bound_context.function_call_options.as_attrs(),
        261     )


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/context.py:1486, in Context.call_function(self, name, tensor_inputs, num_outputs)
       1484 cancellation_context = cancellation.context()
       1485 if cancellation_context is None:
    -> 1486   outputs = execute.execute(
       1487       name.decode("utf-8"),
       1488       num_outputs=num_outputs,
       1489       inputs=tensor_inputs,
       1490       attrs=attrs,
       1491       ctx=self,
       1492   )
       1493 else:
       1494   outputs = execute.execute_with_cancellation(
       1495       name.decode("utf-8"),
       1496       num_outputs=num_outputs,
       (...)
       1500       cancellation_manager=cancellation_context,
       1501   )


    File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/execute.py:53, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         51 try:
         52   ctx.ensure_initialized()
    ---> 53   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         54                                       inputs, attrs, num_outputs)
         55 except core._NotOkStatusException as e:
         56   if name is not None:


    KeyboardInterrupt: 



```python
model = load_model('keras_model')
```


```python
probas = model.predict(X_valid)
```

    123/123 [==============================] - 166s 1s/step



```python
#Even without using glove word embeddings and only trainng till 4 epochs able to get good f1-score
f1_score(yvalid_encoded, np.argmax(probas,axis = 1), average='weighted')
```




    0.7809269071669251




```python

```
