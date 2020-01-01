---
title: "Data Mining and Cleaning Project-Data From Web"
date: 2018-04-20
tags: [data mining, data science, web data, python, pandas]
excerpt: "data mining, data science, web data, pyhton, pandas"
mathjax: "true"
---



**In this pyhthon note book, showing the methods and techniques for extracting data from web and cleaning it for the purpose of analysis or machine learning.**


```python
import pandas as pd
```

Load up the table from the website directly and extract the dataset out of it


```python
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')[0]
```

Check the dataframe by examining the first 5 rows...


```python
df.head(5)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>NaN</td>
      <td>PP</td>
      <td>PP</td>
      <td>SH</td>
      <td>SH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RK</td>
      <td>PLAYER</td>
      <td>TEAM</td>
      <td>GP</td>
      <td>G</td>
      <td>A</td>
      <td>PTS</td>
      <td>+/-</td>
      <td>PIM</td>
      <td>PTS/G</td>
      <td>SOG</td>
      <td>PCT</td>
      <td>GWG</td>
      <td>G</td>
      <td>A</td>
      <td>G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Jamie Benn, LW</td>
      <td>DAL</td>
      <td>82</td>
      <td>35</td>
      <td>52</td>
      <td>87</td>
      <td>1</td>
      <td>64</td>
      <td>1.06</td>
      <td>253</td>
      <td>13.8</td>
      <td>6</td>
      <td>10</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>John Tavares, C</td>
      <td>NYI</td>
      <td>82</td>
      <td>38</td>
      <td>48</td>
      <td>86</td>
      <td>5</td>
      <td>46</td>
      <td>1.05</td>
      <td>278</td>
      <td>13.7</td>
      <td>8</td>
      <td>13</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Sidney Crosby, C</td>
      <td>PIT</td>
      <td>77</td>
      <td>28</td>
      <td>56</td>
      <td>84</td>
      <td>5</td>
      <td>47</td>
      <td>1.09</td>
      <td>237</td>
      <td>11.8</td>
      <td>3</td>
      <td>10</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Rename the columns so that they are similar to the column definitions provided to you on the website.


```python
df.columns = ['RK', 'PLAYER', 'TEAM', 'GamesPlayed', 'Goals', 'Assists', 'Points', 'PlusMinusRating', 'PenaltyMinutes', 'PointsPerGame', 'shotsonGoal', 'ShootingPercentage', 'GameWinningGoals', 'PowerPlayGoals', 'PowerPlayAssists', 'ShorHandedGoals', 'ShortHandedAssists' ]
```

check the date set again to see the changes


```python
df.head(5)
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
      <th>RK</th>
      <th>PLAYER</th>
      <th>TEAM</th>
      <th>GamesPlayed</th>
      <th>Goals</th>
      <th>Assists</th>
      <th>Points</th>
      <th>PlusMinusRating</th>
      <th>PenaltyMinutes</th>
      <th>PointsPerGame</th>
      <th>shotsonGoal</th>
      <th>ShootingPercentage</th>
      <th>GameWinningGoals</th>
      <th>PowerPlayGoals</th>
      <th>PowerPlayAssists</th>
      <th>ShorHandedGoals</th>
      <th>ShortHandedAssists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>NaN</td>
      <td>PP</td>
      <td>PP</td>
      <td>SH</td>
      <td>SH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RK</td>
      <td>PLAYER</td>
      <td>TEAM</td>
      <td>GP</td>
      <td>G</td>
      <td>A</td>
      <td>PTS</td>
      <td>+/-</td>
      <td>PIM</td>
      <td>PTS/G</td>
      <td>SOG</td>
      <td>PCT</td>
      <td>GWG</td>
      <td>G</td>
      <td>A</td>
      <td>G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Jamie Benn, LW</td>
      <td>DAL</td>
      <td>82</td>
      <td>35</td>
      <td>52</td>
      <td>87</td>
      <td>1</td>
      <td>64</td>
      <td>1.06</td>
      <td>253</td>
      <td>13.8</td>
      <td>6</td>
      <td>10</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>John Tavares, C</td>
      <td>NYI</td>
      <td>82</td>
      <td>38</td>
      <td>48</td>
      <td>86</td>
      <td>5</td>
      <td>46</td>
      <td>1.05</td>
      <td>278</td>
      <td>13.7</td>
      <td>8</td>
      <td>13</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Sidney Crosby, C</td>
      <td>PIT</td>
      <td>77</td>
      <td>28</td>
      <td>56</td>
      <td>84</td>
      <td>5</td>
      <td>47</td>
      <td>1.09</td>
      <td>237</td>
      <td>11.8</td>
      <td>3</td>
      <td>10</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Get rid of any row that has at least 4 NANs in it,e.g. that do not contain player points statistics


```python
df.dropna(thresh= len(df.columns)-4, axis=1)
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
      <th>RK</th>
      <th>PLAYER</th>
      <th>TEAM</th>
      <th>GamesPlayed</th>
      <th>Goals</th>
      <th>Assists</th>
      <th>Points</th>
      <th>PlusMinusRating</th>
      <th>PenaltyMinutes</th>
      <th>PointsPerGame</th>
      <th>shotsonGoal</th>
      <th>ShootingPercentage</th>
      <th>GameWinningGoals</th>
      <th>PowerPlayGoals</th>
      <th>PowerPlayAssists</th>
      <th>ShorHandedGoals</th>
      <th>ShortHandedAssists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>NaN</td>
      <td>PP</td>
      <td>PP</td>
      <td>SH</td>
      <td>SH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RK</td>
      <td>PLAYER</td>
      <td>TEAM</td>
      <td>GP</td>
      <td>G</td>
      <td>A</td>
      <td>PTS</td>
      <td>+/-</td>
      <td>PIM</td>
      <td>PTS/G</td>
      <td>SOG</td>
      <td>PCT</td>
      <td>GWG</td>
      <td>G</td>
      <td>A</td>
      <td>G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Jamie Benn, LW</td>
      <td>DAL</td>
      <td>82</td>
      <td>35</td>
      <td>52</td>
      <td>87</td>
      <td>1</td>
      <td>64</td>
      <td>1.06</td>
      <td>253</td>
      <td>13.8</td>
      <td>6</td>
      <td>10</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>John Tavares, C</td>
      <td>NYI</td>
      <td>82</td>
      <td>38</td>
      <td>48</td>
      <td>86</td>
      <td>5</td>
      <td>46</td>
      <td>1.05</td>
      <td>278</td>
      <td>13.7</td>
      <td>8</td>
      <td>13</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Sidney Crosby, C</td>
      <td>PIT</td>
      <td>77</td>
      <td>28</td>
      <td>56</td>
      <td>84</td>
      <td>5</td>
      <td>47</td>
      <td>1.09</td>
      <td>237</td>
      <td>11.8</td>
      <td>3</td>
      <td>10</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>Alex Ovechkin, LW</td>
      <td>WSH</td>
      <td>81</td>
      <td>53</td>
      <td>28</td>
      <td>81</td>
      <td>10</td>
      <td>58</td>
      <td>1.00</td>
      <td>395</td>
      <td>13.4</td>
      <td>11</td>
      <td>25</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>Jakub Voracek, RW</td>
      <td>PHI</td>
      <td>82</td>
      <td>22</td>
      <td>59</td>
      <td>81</td>
      <td>1</td>
      <td>78</td>
      <td>0.99</td>
      <td>221</td>
      <td>10.0</td>
      <td>3</td>
      <td>11</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>Nicklas Backstrom, C</td>
      <td>WSH</td>
      <td>82</td>
      <td>18</td>
      <td>60</td>
      <td>78</td>
      <td>5</td>
      <td>40</td>
      <td>0.95</td>
      <td>153</td>
      <td>11.8</td>
      <td>3</td>
      <td>3</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
      <td>Tyler Seguin, C</td>
      <td>DAL</td>
      <td>71</td>
      <td>37</td>
      <td>40</td>
      <td>77</td>
      <td>-1</td>
      <td>20</td>
      <td>1.08</td>
      <td>280</td>
      <td>13.2</td>
      <td>5</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>Jiri Hudler, LW</td>
      <td>CGY</td>
      <td>78</td>
      <td>31</td>
      <td>45</td>
      <td>76</td>
      <td>17</td>
      <td>14</td>
      <td>0.97</td>
      <td>158</td>
      <td>19.6</td>
      <td>5</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>Daniel Sedin, LW</td>
      <td>VAN</td>
      <td>82</td>
      <td>20</td>
      <td>56</td>
      <td>76</td>
      <td>5</td>
      <td>18</td>
      <td>0.93</td>
      <td>226</td>
      <td>8.9</td>
      <td>5</td>
      <td>4</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10</td>
      <td>Vladimir Tarasenko, RW</td>
      <td>STL</td>
      <td>77</td>
      <td>37</td>
      <td>36</td>
      <td>73</td>
      <td>27</td>
      <td>31</td>
      <td>0.95</td>
      <td>264</td>
      <td>14.0</td>
      <td>6</td>
      <td>8</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
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
      <td>NaN</td>
      <td>PP</td>
      <td>PP</td>
      <td>SH</td>
      <td>SH</td>
    </tr>
    <tr>
      <th>13</th>
      <td>RK</td>
      <td>PLAYER</td>
      <td>TEAM</td>
      <td>GP</td>
      <td>G</td>
      <td>A</td>
      <td>PTS</td>
      <td>+/-</td>
      <td>PIM</td>
      <td>PTS/G</td>
      <td>SOG</td>
      <td>PCT</td>
      <td>GWG</td>
      <td>G</td>
      <td>A</td>
      <td>G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NaN</td>
      <td>Nick Foligno, LW</td>
      <td>CBJ</td>
      <td>79</td>
      <td>31</td>
      <td>42</td>
      <td>73</td>
      <td>16</td>
      <td>50</td>
      <td>0.92</td>
      <td>182</td>
      <td>17.0</td>
      <td>3</td>
      <td>11</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NaN</td>
      <td>Claude Giroux, LW</td>
      <td>PHI</td>
      <td>81</td>
      <td>25</td>
      <td>48</td>
      <td>73</td>
      <td>-3</td>
      <td>36</td>
      <td>0.90</td>
      <td>279</td>
      <td>9.0</td>
      <td>4</td>
      <td>14</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NaN</td>
      <td>Henrik Sedin, C</td>
      <td>VAN</td>
      <td>82</td>
      <td>18</td>
      <td>55</td>
      <td>73</td>
      <td>11</td>
      <td>22</td>
      <td>0.89</td>
      <td>101</td>
      <td>17.8</td>
      <td>0</td>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>14</td>
      <td>Steven Stamkos, C</td>
      <td>TB</td>
      <td>82</td>
      <td>43</td>
      <td>29</td>
      <td>72</td>
      <td>2</td>
      <td>49</td>
      <td>0.88</td>
      <td>268</td>
      <td>16.0</td>
      <td>6</td>
      <td>13</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NaN</td>
      <td>Tyler Johnson, C</td>
      <td>TB</td>
      <td>77</td>
      <td>29</td>
      <td>43</td>
      <td>72</td>
      <td>33</td>
      <td>24</td>
      <td>0.94</td>
      <td>203</td>
      <td>14.3</td>
      <td>6</td>
      <td>8</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>16</td>
      <td>Ryan Johansen, C</td>
      <td>CBJ</td>
      <td>82</td>
      <td>26</td>
      <td>45</td>
      <td>71</td>
      <td>-6</td>
      <td>40</td>
      <td>0.87</td>
      <td>202</td>
      <td>12.9</td>
      <td>0</td>
      <td>7</td>
      <td>19</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>17</td>
      <td>Joe Pavelski, C</td>
      <td>SJ</td>
      <td>82</td>
      <td>37</td>
      <td>33</td>
      <td>70</td>
      <td>12</td>
      <td>29</td>
      <td>0.85</td>
      <td>261</td>
      <td>14.2</td>
      <td>5</td>
      <td>19</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>NaN</td>
      <td>Evgeni Malkin, C</td>
      <td>PIT</td>
      <td>69</td>
      <td>28</td>
      <td>42</td>
      <td>70</td>
      <td>-2</td>
      <td>60</td>
      <td>1.01</td>
      <td>212</td>
      <td>13.2</td>
      <td>4</td>
      <td>9</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NaN</td>
      <td>Ryan Getzlaf, C</td>
      <td>ANA</td>
      <td>77</td>
      <td>25</td>
      <td>45</td>
      <td>70</td>
      <td>15</td>
      <td>62</td>
      <td>0.91</td>
      <td>191</td>
      <td>13.1</td>
      <td>6</td>
      <td>3</td>
      <td>10</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>20</td>
      <td>Rick Nash, LW</td>
      <td>NYR</td>
      <td>79</td>
      <td>42</td>
      <td>27</td>
      <td>69</td>
      <td>29</td>
      <td>36</td>
      <td>0.87</td>
      <td>304</td>
      <td>13.8</td>
      <td>8</td>
      <td>6</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
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
      <td>NaN</td>
      <td>PP</td>
      <td>PP</td>
      <td>SH</td>
      <td>SH</td>
    </tr>
    <tr>
      <th>25</th>
      <td>RK</td>
      <td>PLAYER</td>
      <td>TEAM</td>
      <td>GP</td>
      <td>G</td>
      <td>A</td>
      <td>PTS</td>
      <td>+/-</td>
      <td>PIM</td>
      <td>PTS/G</td>
      <td>SOG</td>
      <td>PCT</td>
      <td>GWG</td>
      <td>G</td>
      <td>A</td>
      <td>G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>26</th>
      <td>21</td>
      <td>Max Pacioretty, LW</td>
      <td>MTL</td>
      <td>80</td>
      <td>37</td>
      <td>30</td>
      <td>67</td>
      <td>38</td>
      <td>32</td>
      <td>0.84</td>
      <td>302</td>
      <td>12.3</td>
      <td>10</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>Logan Couture, C</td>
      <td>SJ</td>
      <td>82</td>
      <td>27</td>
      <td>40</td>
      <td>67</td>
      <td>-6</td>
      <td>12</td>
      <td>0.82</td>
      <td>263</td>
      <td>10.3</td>
      <td>4</td>
      <td>6</td>
      <td>18</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>23</td>
      <td>Jonathan Toews, C</td>
      <td>CHI</td>
      <td>81</td>
      <td>28</td>
      <td>38</td>
      <td>66</td>
      <td>30</td>
      <td>36</td>
      <td>0.81</td>
      <td>192</td>
      <td>14.6</td>
      <td>7</td>
      <td>6</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>Erik Karlsson, D</td>
      <td>OTT</td>
      <td>82</td>
      <td>21</td>
      <td>45</td>
      <td>66</td>
      <td>7</td>
      <td>42</td>
      <td>0.80</td>
      <td>292</td>
      <td>7.2</td>
      <td>3</td>
      <td>6</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>NaN</td>
      <td>Henrik Zetterberg, LW</td>
      <td>DET</td>
      <td>77</td>
      <td>17</td>
      <td>49</td>
      <td>66</td>
      <td>-6</td>
      <td>32</td>
      <td>0.86</td>
      <td>227</td>
      <td>7.5</td>
      <td>3</td>
      <td>4</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>26</td>
      <td>Pavel Datsyuk, C</td>
      <td>DET</td>
      <td>63</td>
      <td>26</td>
      <td>39</td>
      <td>65</td>
      <td>12</td>
      <td>8</td>
      <td>1.03</td>
      <td>165</td>
      <td>15.8</td>
      <td>5</td>
      <td>8</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NaN</td>
      <td>Joe Thornton, C</td>
      <td>SJ</td>
      <td>78</td>
      <td>16</td>
      <td>49</td>
      <td>65</td>
      <td>-4</td>
      <td>30</td>
      <td>0.83</td>
      <td>131</td>
      <td>12.2</td>
      <td>0</td>
      <td>4</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>28</td>
      <td>Nikita Kucherov, RW</td>
      <td>TB</td>
      <td>82</td>
      <td>28</td>
      <td>36</td>
      <td>64</td>
      <td>38</td>
      <td>37</td>
      <td>0.78</td>
      <td>190</td>
      <td>14.7</td>
      <td>2</td>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>NaN</td>
      <td>Patrick Kane, RW</td>
      <td>CHI</td>
      <td>61</td>
      <td>27</td>
      <td>37</td>
      <td>64</td>
      <td>10</td>
      <td>10</td>
      <td>1.05</td>
      <td>186</td>
      <td>14.5</td>
      <td>5</td>
      <td>6</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>NaN</td>
      <td>Mark Stone, RW</td>
      <td>OTT</td>
      <td>80</td>
      <td>26</td>
      <td>38</td>
      <td>64</td>
      <td>21</td>
      <td>14</td>
      <td>0.80</td>
      <td>157</td>
      <td>16.6</td>
      <td>6</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
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
      <td>NaN</td>
      <td>PP</td>
      <td>PP</td>
      <td>SH</td>
      <td>SH</td>
    </tr>
    <tr>
      <th>37</th>
      <td>RK</td>
      <td>PLAYER</td>
      <td>TEAM</td>
      <td>GP</td>
      <td>G</td>
      <td>A</td>
      <td>PTS</td>
      <td>+/-</td>
      <td>PIM</td>
      <td>PTS/G</td>
      <td>SOG</td>
      <td>PCT</td>
      <td>GWG</td>
      <td>G</td>
      <td>A</td>
      <td>G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NaN</td>
      <td>Kyle Turris, C</td>
      <td>OTT</td>
      <td>82</td>
      <td>24</td>
      <td>40</td>
      <td>64</td>
      <td>5</td>
      <td>36</td>
      <td>0.78</td>
      <td>215</td>
      <td>11.2</td>
      <td>6</td>
      <td>4</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NaN</td>
      <td>Alexander Steen, LW</td>
      <td>STL</td>
      <td>74</td>
      <td>24</td>
      <td>40</td>
      <td>64</td>
      <td>8</td>
      <td>33</td>
      <td>0.86</td>
      <td>223</td>
      <td>10.8</td>
      <td>5</td>
      <td>8</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NaN</td>
      <td>Johnny Gaudreau, LW</td>
      <td>CGY</td>
      <td>80</td>
      <td>24</td>
      <td>40</td>
      <td>64</td>
      <td>11</td>
      <td>14</td>
      <td>0.80</td>
      <td>167</td>
      <td>14.4</td>
      <td>4</td>
      <td>8</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>NaN</td>
      <td>Anze Kopitar, C</td>
      <td>LA</td>
      <td>79</td>
      <td>16</td>
      <td>48</td>
      <td>64</td>
      <td>-2</td>
      <td>10</td>
      <td>0.81</td>
      <td>134</td>
      <td>11.9</td>
      <td>4</td>
      <td>6</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>35</td>
      <td>Radim Vrbata, RW</td>
      <td>VAN</td>
      <td>79</td>
      <td>31</td>
      <td>32</td>
      <td>63</td>
      <td>6</td>
      <td>20</td>
      <td>0.80</td>
      <td>267</td>
      <td>11.6</td>
      <td>7</td>
      <td>12</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>NaN</td>
      <td>Jaden Schwartz, LW</td>
      <td>STL</td>
      <td>75</td>
      <td>28</td>
      <td>35</td>
      <td>63</td>
      <td>13</td>
      <td>16</td>
      <td>0.84</td>
      <td>184</td>
      <td>15.2</td>
      <td>4</td>
      <td>8</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>44</th>
      <td>NaN</td>
      <td>Filip Forsberg, LW</td>
      <td>NSH</td>
      <td>82</td>
      <td>26</td>
      <td>37</td>
      <td>63</td>
      <td>15</td>
      <td>24</td>
      <td>0.77</td>
      <td>237</td>
      <td>11.0</td>
      <td>6</td>
      <td>6</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>NaN</td>
      <td>Jordan Eberle, RW</td>
      <td>EDM</td>
      <td>81</td>
      <td>24</td>
      <td>39</td>
      <td>63</td>
      <td>-16</td>
      <td>24</td>
      <td>0.78</td>
      <td>183</td>
      <td>13.1</td>
      <td>2</td>
      <td>6</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>NaN</td>
      <td>Ondrej Palat, LW</td>
      <td>TB</td>
      <td>75</td>
      <td>16</td>
      <td>47</td>
      <td>63</td>
      <td>31</td>
      <td>24</td>
      <td>0.84</td>
      <td>139</td>
      <td>11.5</td>
      <td>5</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>40</td>
      <td>Zach Parise, LW</td>
      <td>MIN</td>
      <td>74</td>
      <td>33</td>
      <td>29</td>
      <td>62</td>
      <td>21</td>
      <td>41</td>
      <td>0.84</td>
      <td>259</td>
      <td>12.7</td>
      <td>3</td>
      <td>11</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let us see What indexing command(s) can you use to select all rows EXCEPT those rows


```python
df.drop([0, 1, 12, 13, 24, 25, 36, 37], axis = 0, inplace = True)
```

Let us see the data set with this changes


```python
df.shape
```




    (40, 17)




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
      <th>RK</th>
      <th>PLAYER</th>
      <th>TEAM</th>
      <th>GamesPlayed</th>
      <th>Goals</th>
      <th>Assists</th>
      <th>Points</th>
      <th>PlusMinusRating</th>
      <th>PenaltyMinutes</th>
      <th>PointsPerGame</th>
      <th>shotsonGoal</th>
      <th>ShootingPercentage</th>
      <th>GameWinningGoals</th>
      <th>PowerPlayGoals</th>
      <th>PowerPlayAssists</th>
      <th>ShorHandedGoals</th>
      <th>ShortHandedAssists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Jamie Benn, LW</td>
      <td>DAL</td>
      <td>82</td>
      <td>35</td>
      <td>52</td>
      <td>87</td>
      <td>1</td>
      <td>64</td>
      <td>1.06</td>
      <td>253</td>
      <td>13.8</td>
      <td>6</td>
      <td>10</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>John Tavares, C</td>
      <td>NYI</td>
      <td>82</td>
      <td>38</td>
      <td>48</td>
      <td>86</td>
      <td>5</td>
      <td>46</td>
      <td>1.05</td>
      <td>278</td>
      <td>13.7</td>
      <td>8</td>
      <td>13</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Sidney Crosby, C</td>
      <td>PIT</td>
      <td>77</td>
      <td>28</td>
      <td>56</td>
      <td>84</td>
      <td>5</td>
      <td>47</td>
      <td>1.09</td>
      <td>237</td>
      <td>11.8</td>
      <td>3</td>
      <td>10</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>Alex Ovechkin, LW</td>
      <td>WSH</td>
      <td>81</td>
      <td>53</td>
      <td>28</td>
      <td>81</td>
      <td>10</td>
      <td>58</td>
      <td>1.00</td>
      <td>395</td>
      <td>13.4</td>
      <td>11</td>
      <td>25</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>Jakub Voracek, RW</td>
      <td>PHI</td>
      <td>82</td>
      <td>22</td>
      <td>59</td>
      <td>81</td>
      <td>1</td>
      <td>78</td>
      <td>0.99</td>
      <td>221</td>
      <td>10.0</td>
      <td>3</td>
      <td>11</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Get rid of the 'RK' column


```python
df.drop('RK', axis = 1,inplace = True)
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
      <th>PLAYER</th>
      <th>TEAM</th>
      <th>GamesPlayed</th>
      <th>Goals</th>
      <th>Assists</th>
      <th>Points</th>
      <th>PlusMinusRating</th>
      <th>PenaltyMinutes</th>
      <th>PointsPerGame</th>
      <th>shotsonGoal</th>
      <th>ShootingPercentage</th>
      <th>GameWinningGoals</th>
      <th>PowerPlayGoals</th>
      <th>PowerPlayAssists</th>
      <th>ShorHandedGoals</th>
      <th>ShortHandedAssists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Jamie Benn, LW</td>
      <td>DAL</td>
      <td>82</td>
      <td>35</td>
      <td>52</td>
      <td>87</td>
      <td>1</td>
      <td>64</td>
      <td>1.06</td>
      <td>253</td>
      <td>13.8</td>
      <td>6</td>
      <td>10</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>John Tavares, C</td>
      <td>NYI</td>
      <td>82</td>
      <td>38</td>
      <td>48</td>
      <td>86</td>
      <td>5</td>
      <td>46</td>
      <td>1.05</td>
      <td>278</td>
      <td>13.7</td>
      <td>8</td>
      <td>13</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sidney Crosby, C</td>
      <td>PIT</td>
      <td>77</td>
      <td>28</td>
      <td>56</td>
      <td>84</td>
      <td>5</td>
      <td>47</td>
      <td>1.09</td>
      <td>237</td>
      <td>11.8</td>
      <td>3</td>
      <td>10</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Alex Ovechkin, LW</td>
      <td>WSH</td>
      <td>81</td>
      <td>53</td>
      <td>28</td>
      <td>81</td>
      <td>10</td>
      <td>58</td>
      <td>1.00</td>
      <td>395</td>
      <td>13.4</td>
      <td>11</td>
      <td>25</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jakub Voracek, RW</td>
      <td>PHI</td>
      <td>82</td>
      <td>22</td>
      <td>59</td>
      <td>81</td>
      <td>1</td>
      <td>78</td>
      <td>0.99</td>
      <td>221</td>
      <td>10.0</td>
      <td>3</td>
      <td>11</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Ensure there are no holes in your index by resetting it. By the way, don't store the original index


```python
df.reset_index(inplace = True, drop = True)
```

ensure those that all columns should be numeric are numeric


```python
df.GamesPlayed  = pd.to_numeric(df.GamesPlayed)       
df.Goals         = pd.to_numeric(df.GamesPlayed)
df.Assists        = pd.to_numeric(df.GamesPlayed)
df.Points         = pd.to_numeric(df.GamesPlayed)
df.PlusMinusRating  = pd.to_numeric(df.GamesPlayed)
df.PenaltyMinutes  = pd.to_numeric(df.GamesPlayed)
df.PointsPerGame    = pd.to_numeric(df.GamesPlayed)
df.shotsonGoal      = pd.to_numeric(df.GamesPlayed)
df.ShootingPercentage = pd.to_numeric(df.GamesPlayed)
df.GameWinningGoals  = pd.to_numeric(df.GamesPlayed)
df.PowerPlayGoals    = pd.to_numeric(df.GamesPlayed)
df.PowerPlayAssists    = pd.to_numeric(df.GamesPlayed)
df.ShorHandedGoals    = pd.to_numeric(df.GamesPlayed)
df.ShortHandedAssists   = pd.to_numeric(df.GamesPlayed)
```


```python
df.dtypes
```




    PLAYER                object
    TEAM                  object
    GamesPlayed            int64
    Goals                  int64
    Assists                int64
    Points                 int64
    PlusMinusRating        int64
    PenaltyMinutes         int64
    PointsPerGame          int64
    shotsonGoal            int64
    ShootingPercentage     int64
    GameWinningGoals       int64
    PowerPlayGoals         int64
    PowerPlayAssists       int64
    ShorHandedGoals        int64
    ShortHandedAssists     int64
    dtype: object



Your dataframe is now ready for exploaration! See the final dataframe!!!!!!!


```python
df.shape
```




    (40, 16)




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
      <th>PLAYER</th>
      <th>TEAM</th>
      <th>GamesPlayed</th>
      <th>Goals</th>
      <th>Assists</th>
      <th>Points</th>
      <th>PlusMinusRating</th>
      <th>PenaltyMinutes</th>
      <th>PointsPerGame</th>
      <th>shotsonGoal</th>
      <th>ShootingPercentage</th>
      <th>GameWinningGoals</th>
      <th>PowerPlayGoals</th>
      <th>PowerPlayAssists</th>
      <th>ShorHandedGoals</th>
      <th>ShortHandedAssists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jamie Benn, LW</td>
      <td>DAL</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>John Tavares, C</td>
      <td>NYI</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sidney Crosby, C</td>
      <td>PIT</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alex Ovechkin, LW</td>
      <td>WSH</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jakub Voracek, RW</td>
      <td>PHI</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
