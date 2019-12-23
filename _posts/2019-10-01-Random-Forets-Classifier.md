---
title: "Forest Type Classification Project"
date: 2019-10-01
tags: [data wrangling, data science, machine learning model, Random Forest Classifier]
header:
  image:"/images/randomforest/Figure-2-Random-Forest-Predictions.png"
excerpt: "data wrangling, data science, machine learning model, Random Forest Classifier"
mathjax: "true"
---

## Random Forest Classifier 
**Classify forest types based on information about the area**

## Project Description
This is a competition project hosted on [Kaggle.com](www.Kaggle.com)

In this project we’ll predict what types of trees there are in an area based on various geographic features.

## Data Set
The datasets comes from a study conducted in four wilderness areas within the beautiful Roosevelt National Forest of northern Colorado. These areas represent forests with very little human disturbances – the existing forest cover types there are more a result of ecological processes rather than forest management practices.
The data is in raw form and contains categorical data such as wilderness areas and soil type.

Acknowledgements:
This dataset was provided by Jock A. Blackard and Colorado State University. We also thank the UCI machine learning repository for hosting the dataset. 
Data set can be downloaded from this [link](https://www.kaggle.com/c/learn-together/data)

## Evaluation
Models are evaluated on categorization accuracy.

## Basic Random Forets Classifier
**The challenge:**

In this competition you’ll predict what types of trees there are in an area based on various geographic features.

The competition datasets comes from a study conducted in four wilderness areas within the beautiful Roosevelt National Forest of northern Colorado. These areas represent forests with very little human disturbances – the existing forest cover types there are more a result of ecological processes rather than forest management practices.
The data is in raw form and contains categorical data such as wilderness areas and soil type.

### Import Required Libraries


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2 # For feature selection

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/learn-together/train.csv
    /kaggle/input/learn-together/sample_submission.csv
    /kaggle/input/learn-together/test.csv



```python
!pwd
```

    /kaggle/working


### Load Data Set


```python
print("Loading data set......")
train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")
print("Done...")
```

    Loading data set......
    Done...


### Exploratory Data Analysis


```python
print("train data size:", train.shape)
train.head()
```

    train data size: (15120, 56)





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
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>...</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 56 columns</p>
</div>




```python
print("test data size:", test.shape)
test.head()
```

    test data size: (565892, 55)





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
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>...</th>
      <th>Soil_Type31</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>15121</td>
      <td>2680</td>
      <td>354</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>2684</td>
      <td>196</td>
      <td>214</td>
      <td>156</td>
      <td>...</td>
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
      <td>1</td>
      <td>15122</td>
      <td>2683</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>2654</td>
      <td>201</td>
      <td>216</td>
      <td>152</td>
      <td>...</td>
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
      <td>2</td>
      <td>15123</td>
      <td>2713</td>
      <td>16</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>2980</td>
      <td>206</td>
      <td>208</td>
      <td>137</td>
      <td>...</td>
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
      <td>3</td>
      <td>15124</td>
      <td>2709</td>
      <td>24</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>2950</td>
      <td>208</td>
      <td>201</td>
      <td>125</td>
      <td>...</td>
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
      <td>4</td>
      <td>15125</td>
      <td>2706</td>
      <td>29</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>2920</td>
      <td>210</td>
      <td>195</td>
      <td>115</td>
      <td>...</td>
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
<p>5 rows × 55 columns</p>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15120 entries, 0 to 15119
    Data columns (total 56 columns):
    Id                                    15120 non-null int64
    Elevation                             15120 non-null int64
    Aspect                                15120 non-null int64
    Slope                                 15120 non-null int64
    Horizontal_Distance_To_Hydrology      15120 non-null int64
    Vertical_Distance_To_Hydrology        15120 non-null int64
    Horizontal_Distance_To_Roadways       15120 non-null int64
    Hillshade_9am                         15120 non-null int64
    Hillshade_Noon                        15120 non-null int64
    Hillshade_3pm                         15120 non-null int64
    Horizontal_Distance_To_Fire_Points    15120 non-null int64
    Wilderness_Area1                      15120 non-null int64
    Wilderness_Area2                      15120 non-null int64
    Wilderness_Area3                      15120 non-null int64
    Wilderness_Area4                      15120 non-null int64
    Soil_Type1                            15120 non-null int64
    Soil_Type2                            15120 non-null int64
    Soil_Type3                            15120 non-null int64
    Soil_Type4                            15120 non-null int64
    Soil_Type5                            15120 non-null int64
    Soil_Type6                            15120 non-null int64
    Soil_Type7                            15120 non-null int64
    Soil_Type8                            15120 non-null int64
    Soil_Type9                            15120 non-null int64
    Soil_Type10                           15120 non-null int64
    Soil_Type11                           15120 non-null int64
    Soil_Type12                           15120 non-null int64
    Soil_Type13                           15120 non-null int64
    Soil_Type14                           15120 non-null int64
    Soil_Type15                           15120 non-null int64
    Soil_Type16                           15120 non-null int64
    Soil_Type17                           15120 non-null int64
    Soil_Type18                           15120 non-null int64
    Soil_Type19                           15120 non-null int64
    Soil_Type20                           15120 non-null int64
    Soil_Type21                           15120 non-null int64
    Soil_Type22                           15120 non-null int64
    Soil_Type23                           15120 non-null int64
    Soil_Type24                           15120 non-null int64
    Soil_Type25                           15120 non-null int64
    Soil_Type26                           15120 non-null int64
    Soil_Type27                           15120 non-null int64
    Soil_Type28                           15120 non-null int64
    Soil_Type29                           15120 non-null int64
    Soil_Type30                           15120 non-null int64
    Soil_Type31                           15120 non-null int64
    Soil_Type32                           15120 non-null int64
    Soil_Type33                           15120 non-null int64
    Soil_Type34                           15120 non-null int64
    Soil_Type35                           15120 non-null int64
    Soil_Type36                           15120 non-null int64
    Soil_Type37                           15120 non-null int64
    Soil_Type38                           15120 non-null int64
    Soil_Type39                           15120 non-null int64
    Soil_Type40                           15120 non-null int64
    Cover_Type                            15120 non-null int64
    dtypes: int64(56)
    memory usage: 6.5 MB



```python
train.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Id</td>
      <td>15120.0</td>
      <td>7560.500000</td>
      <td>4364.912370</td>
      <td>1.0</td>
      <td>3780.75</td>
      <td>7560.5</td>
      <td>11340.25</td>
      <td>15120.0</td>
    </tr>
    <tr>
      <td>Elevation</td>
      <td>15120.0</td>
      <td>2749.322553</td>
      <td>417.678187</td>
      <td>1863.0</td>
      <td>2376.00</td>
      <td>2752.0</td>
      <td>3104.00</td>
      <td>3849.0</td>
    </tr>
    <tr>
      <td>Aspect</td>
      <td>15120.0</td>
      <td>156.676653</td>
      <td>110.085801</td>
      <td>0.0</td>
      <td>65.00</td>
      <td>126.0</td>
      <td>261.00</td>
      <td>360.0</td>
    </tr>
    <tr>
      <td>Slope</td>
      <td>15120.0</td>
      <td>16.501587</td>
      <td>8.453927</td>
      <td>0.0</td>
      <td>10.00</td>
      <td>15.0</td>
      <td>22.00</td>
      <td>52.0</td>
    </tr>
    <tr>
      <td>Horizontal_Distance_To_Hydrology</td>
      <td>15120.0</td>
      <td>227.195701</td>
      <td>210.075296</td>
      <td>0.0</td>
      <td>67.00</td>
      <td>180.0</td>
      <td>330.00</td>
      <td>1343.0</td>
    </tr>
    <tr>
      <td>Vertical_Distance_To_Hydrology</td>
      <td>15120.0</td>
      <td>51.076521</td>
      <td>61.239406</td>
      <td>-146.0</td>
      <td>5.00</td>
      <td>32.0</td>
      <td>79.00</td>
      <td>554.0</td>
    </tr>
    <tr>
      <td>Horizontal_Distance_To_Roadways</td>
      <td>15120.0</td>
      <td>1714.023214</td>
      <td>1325.066358</td>
      <td>0.0</td>
      <td>764.00</td>
      <td>1316.0</td>
      <td>2270.00</td>
      <td>6890.0</td>
    </tr>
    <tr>
      <td>Hillshade_9am</td>
      <td>15120.0</td>
      <td>212.704299</td>
      <td>30.561287</td>
      <td>0.0</td>
      <td>196.00</td>
      <td>220.0</td>
      <td>235.00</td>
      <td>254.0</td>
    </tr>
    <tr>
      <td>Hillshade_Noon</td>
      <td>15120.0</td>
      <td>218.965608</td>
      <td>22.801966</td>
      <td>99.0</td>
      <td>207.00</td>
      <td>223.0</td>
      <td>235.00</td>
      <td>254.0</td>
    </tr>
    <tr>
      <td>Hillshade_3pm</td>
      <td>15120.0</td>
      <td>135.091997</td>
      <td>45.895189</td>
      <td>0.0</td>
      <td>106.00</td>
      <td>138.0</td>
      <td>167.00</td>
      <td>248.0</td>
    </tr>
    <tr>
      <td>Horizontal_Distance_To_Fire_Points</td>
      <td>15120.0</td>
      <td>1511.147288</td>
      <td>1099.936493</td>
      <td>0.0</td>
      <td>730.00</td>
      <td>1256.0</td>
      <td>1988.25</td>
      <td>6993.0</td>
    </tr>
    <tr>
      <td>Wilderness_Area1</td>
      <td>15120.0</td>
      <td>0.237897</td>
      <td>0.425810</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Wilderness_Area2</td>
      <td>15120.0</td>
      <td>0.033003</td>
      <td>0.178649</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Wilderness_Area3</td>
      <td>15120.0</td>
      <td>0.419907</td>
      <td>0.493560</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Wilderness_Area4</td>
      <td>15120.0</td>
      <td>0.309193</td>
      <td>0.462176</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type1</td>
      <td>15120.0</td>
      <td>0.023479</td>
      <td>0.151424</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type2</td>
      <td>15120.0</td>
      <td>0.041204</td>
      <td>0.198768</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type3</td>
      <td>15120.0</td>
      <td>0.063624</td>
      <td>0.244091</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type4</td>
      <td>15120.0</td>
      <td>0.055754</td>
      <td>0.229454</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type5</td>
      <td>15120.0</td>
      <td>0.010913</td>
      <td>0.103896</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type6</td>
      <td>15120.0</td>
      <td>0.042989</td>
      <td>0.202840</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type7</td>
      <td>15120.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>Soil_Type8</td>
      <td>15120.0</td>
      <td>0.000066</td>
      <td>0.008133</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type9</td>
      <td>15120.0</td>
      <td>0.000661</td>
      <td>0.025710</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type10</td>
      <td>15120.0</td>
      <td>0.141667</td>
      <td>0.348719</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type11</td>
      <td>15120.0</td>
      <td>0.026852</td>
      <td>0.161656</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type12</td>
      <td>15120.0</td>
      <td>0.015013</td>
      <td>0.121609</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type13</td>
      <td>15120.0</td>
      <td>0.031481</td>
      <td>0.174621</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type14</td>
      <td>15120.0</td>
      <td>0.011177</td>
      <td>0.105133</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type15</td>
      <td>15120.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>Soil_Type16</td>
      <td>15120.0</td>
      <td>0.007540</td>
      <td>0.086506</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type17</td>
      <td>15120.0</td>
      <td>0.040476</td>
      <td>0.197080</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type18</td>
      <td>15120.0</td>
      <td>0.003968</td>
      <td>0.062871</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type19</td>
      <td>15120.0</td>
      <td>0.003042</td>
      <td>0.055075</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type20</td>
      <td>15120.0</td>
      <td>0.009193</td>
      <td>0.095442</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type21</td>
      <td>15120.0</td>
      <td>0.001058</td>
      <td>0.032514</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type22</td>
      <td>15120.0</td>
      <td>0.022817</td>
      <td>0.149326</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type23</td>
      <td>15120.0</td>
      <td>0.050066</td>
      <td>0.218089</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type24</td>
      <td>15120.0</td>
      <td>0.016997</td>
      <td>0.129265</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type25</td>
      <td>15120.0</td>
      <td>0.000066</td>
      <td>0.008133</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type26</td>
      <td>15120.0</td>
      <td>0.003571</td>
      <td>0.059657</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type27</td>
      <td>15120.0</td>
      <td>0.000992</td>
      <td>0.031482</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type28</td>
      <td>15120.0</td>
      <td>0.000595</td>
      <td>0.024391</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type29</td>
      <td>15120.0</td>
      <td>0.085384</td>
      <td>0.279461</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type30</td>
      <td>15120.0</td>
      <td>0.047950</td>
      <td>0.213667</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type31</td>
      <td>15120.0</td>
      <td>0.021958</td>
      <td>0.146550</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type32</td>
      <td>15120.0</td>
      <td>0.045635</td>
      <td>0.208699</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type33</td>
      <td>15120.0</td>
      <td>0.040741</td>
      <td>0.197696</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type34</td>
      <td>15120.0</td>
      <td>0.001455</td>
      <td>0.038118</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type35</td>
      <td>15120.0</td>
      <td>0.006746</td>
      <td>0.081859</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type36</td>
      <td>15120.0</td>
      <td>0.000661</td>
      <td>0.025710</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type37</td>
      <td>15120.0</td>
      <td>0.002249</td>
      <td>0.047368</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type38</td>
      <td>15120.0</td>
      <td>0.048148</td>
      <td>0.214086</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type39</td>
      <td>15120.0</td>
      <td>0.043452</td>
      <td>0.203880</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Soil_Type40</td>
      <td>15120.0</td>
      <td>0.030357</td>
      <td>0.171574</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Cover_Type</td>
      <td>15120.0</td>
      <td>4.000000</td>
      <td>2.000066</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>4.0</td>
      <td>6.00</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Selection

In this model, we are not going to make much feature engineering. We will make model by using the available features. There are some negative values in "Vertical_Distance_To_Hydrology" column. SO we will drop this column along with "Id" column. "Id"column is not required for our model.


```python
# Declare target and predictors
print("Selecting features and target columns for model")
target = train['Cover_Type']
train_df = train.drop(["Cover_Type", "Id", "Vertical_Distance_To_Hydrology"], axis=1)
test_df = test.drop(["Id", "Vertical_Distance_To_Hydrology"], axis=1)

```

    Selecting features and target columns for model



```python
train_df.shape, test_df.shape
```




    ((15120, 53), (565892, 53))



To select best features for our model, we wil use "SelectKBest" module from sklearn feature selection.
Since our target column is categorical, we will use "chi2" as parameter for "SelectKbest".


```python
# Feature  selection

best = SelectKBest(chi2, k=25).fit(train_df, target)
train_best = best.transform(train_df)
test_best = best.transform(test_df)
```

### Create Model

We are using RandomForestClassifier algorithm for our model


```python
# Create Model
print("Creating model")
rf = RandomForestClassifier(n_estimators=100)
print("Model created")

```

    Creating model
    Model created


### Cross validation of the model


```python
print("Cross vaidation Score")
print(cross_val_score(rf,train_best, target, cv=3, scoring="accuracy" ))

```

    Cross vaidation Score
    [0.79424603 0.77083333 0.78650794]


### Fit Model and Predict


```python
print("Fitting Model on training data set.....")
# Fit Model to traing data
rf.fit(train_best, target)
print("Predict on test data set....")
test_pred = rf.predict(test_best)


```

    Fitting Model on training data set.....
    Predict on test data set....


### Create Submission File


```python
# Save test predictions to file
print("Creating submission file")
output = pd.DataFrame({'Id': test.Id,'Cover_Type': test_pred})
output.to_csv('submission_rf_1.csv', index=False)

```

    Creating submission file


### Further work
* More accurate model can be created using features engineering.
* For better performance of this model, Parameter tuning is advised.



```python

```
