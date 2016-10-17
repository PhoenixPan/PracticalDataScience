## Least square regression
We want to know the parameters that make this function the smallest.  
![figure1](https://cloud.githubusercontent.com/assets/14355257/19215995/443cd2f2-8d7b-11e6-915f-747729cd0e20.png)
The function may look like:  
![figure2](https://cloud.githubusercontent.com/assets/14355257/19215997/5212f35c-8d7b-11e6-931e-fe2b46e5a674.png)


## Introduction
One of the most widespread regression tools is the simple but powerful linear regression. In this notebook, you will engineer the Pittsburgh bus data into numerical features and use them to predict the number of minutes until the bus reaches the bus stop at Forbes and Morewood. 

Notebook restriction: you may not use scikit-learn for this notebook.  

## Q1: Labeling the Dataset [8pts]

You may have noticed that the Pittsburgh bus data has a predictions table with the TrueTime predictions on arrival time, however it does not have the true label: the actual number of minutes until a bus reaches Forbes and Morewood. You will have to generate this yourself. 

Using the `all_trips` function that you implemented in homework 2, you can split the dataframe into separate trips. You will first process each trip into a form more natural for the regression setting. For each trip, you will need to locate the point at which a bus passes the bus stop to get the time at which the bus passes the bus stop. From here, you can calculate the true label for all prior datapoints, and throw out the rest. 

### Importing functions from homework 2

Using the menu in Jupyter, you can import code from your notebook as a Python script using the following steps: 
1. Click File -> Download as -> Python (.py)
2. Save file (time_series.py) in the same directory as this notebook 
3. (optional) Remove all test code (i.e. lines between AUTOLAB_IGNORE macros) from the script for faster loading time
4. Import from the notebook with `from time_series import function_name`

### Specifications

1. To determine when the bus passes Morewood, we will use the Euclidean distance as a metric to determine how close the bus is to the bus stop. 
2. We will assume that the row entry with the smallest Euclidean distance to the bus stop is when the bus reaches the bus stop, and that you should truncate all rows that occur **after** this entry.  In the case where there are multiple entries with the exact same minimal distance, you should just consider the first one that occurs in the trip (so truncate everything after the first occurance of minimal distance). 
3. Assume that the row with the smallest Euclidean distance to the bus stop is also the true time at which the bus passes the bus stop. Using this, create a new column called `eta` that contains for each row, the number of minutes until the bus passes the bus stop (so the last row of every trip will have an `eta` of 0).
4. Make sure your `eta` is numerical and not a python timedelta object. 

```
import pandas as pd
import numpy as np
import scipy.linalg as la
from collections import Counter

vdf, _ = load_data('bus_train.db')
all_trips = split_trips(vdf)
```

```
def label_and_truncate(trip, bus_stop_coordinates):
    """ Given a dataframe of a trip following the specification in the previous homework assignment,
        generate the labels and throw away irrelevant rows. 
        
        Args: 
            trip (dataframe): a dataframe from the list outputted by split_trips from homework 2
            stop_coordinates ((float, float)): a pair of floats indicating the (latitude, longitude) 
                                               coordinates of the target bus stop. 
            
        Return:
            (dataframe): a labeled trip that is truncated at Forbes and Morewood and contains a new column 
                         called `eta` which contains the number of minutes until it reaches the bus stop. 
    """
    import math
    count = 0
    morewood_count = 0
    morewood_dist = None
    morewood_time = 0
    eta = []
    for index, row in trip.iterrows():
        count += 1
        temp = math.hypot(row['lat'] - bus_stop_coordinates[0], row['lon'] - bus_stop_coordinates[1])
        if morewood_dist is None or temp < morewood_dist:
            morewood_dist = temp
            morewood_count = count
            morewood_time = index
    result = trip.ix[:morewood_count].copy()
    for index, row in result.iterrows():
        this_eta = (morewood_time - index).seconds / 60
        eta.append(this_eta)
    result['eta'] = pd.Series(eta, index=result.index)
    return result
    
morewood_coordinates = (40.444671114203, -79.94356058465502) # (lat, lon)
# one_trip = label_and_truncate(all_trips[1], morewood_coordinates)
# print one_trip

labeled_trips = [label_and_truncate(trip, morewood_coordinates) for trip in all_trips]
labeled_vdf = pd.concat(labeled_trips).reset_index()
print Counter([len(t) for t in labeled_trips])
print labeled_vdf.head()
```

For our implementation, this returns the following output
```python
>>> Counter([len(t) for t in labeled_trips])
Counter({1: 506, 21: 200, 18: 190, 20: 184, 19: 163, 16: 162, 22: 159, 17: 151, 23: 139, 31: 132, 15: 128, 2: 125, 34: 112, 32: 111, 33: 101, 28: 98, 14: 97, 30: 95, 35: 95, 29: 93, 24: 90, 25: 89, 37: 86, 27: 83, 39: 83, 38: 82, 36: 77, 26: 75, 40: 70, 13: 62, 41: 53, 44: 52, 42: 47, 6: 44, 5: 39, 12: 39, 46: 39, 7: 38, 3: 36, 45: 33, 47: 33, 43: 31, 48: 27, 4: 26, 49: 26, 11: 25, 50: 25, 10: 23, 51: 23, 8: 19, 9: 18, 53: 16, 54: 15, 52: 14, 55: 14, 56: 8, 57: 3, 58: 3, 59: 3, 60: 3, 61: 1, 62: 1, 67: 1}) 
>>> labeled_vdf.head()
               tmstmp   vid        lat        lon  hdg   pid   rt        des  \
0 2016-08-11 10:56:00  5549  40.439504 -79.996981  114  4521  61A  Swissvale   
1 2016-08-11 10:57:00  5549  40.439504 -79.996981  114  4521  61A  Swissvale   
2 2016-08-11 10:58:00  5549  40.438842 -79.994733  124  4521  61A  Swissvale   
3 2016-08-11 10:59:00  5549  40.437938 -79.991213   94  4521  61A  Swissvale   
4 2016-08-11 10:59:00  5549  40.437938 -79.991213   94  4521  61A  Swissvale   

   pdist  spd tablockid  tatripid  eta  
0   1106    0  061A-164      6691   16  
1   1106    0  061A-164      6691   15  
2   1778    8  061A-164      6691   14  
3   2934    7  061A-164      6691   13  
4   2934    7  061A-164      6691   13 
```

## Q2: Generating Basic Features [8pts]
In order to perform linear regression, we need to have numerical features. However, not everything in the bus database is a number, and not all of the numbers even make sense as numerical features. If you use the data as is, it is highly unlikely that you'll achieve anything meaningful.

Consequently, you will perform some basic feature engineering. Feature engineering is extracting "features" or statistics from your data, and hopefully improve the performance if your learning algorithm (in this case, linear regression). Good features can often make up for poor model selection and improve your overall predictive ability on unseen data. In essence, you want to turn your data into something your algorithm understands. 

### Specifications
1. The input to your function will be a concatenation of the trip dataframes generated in Q1 with the index dropped (so same structure as the original dataframe, but with an extra column and less rows). 
2. Linear models typically have a constant bias term. We will encode this as a column of 1s in the dataframe. Call this column 'bias'. 
2. We will keep the following columns as is, since they are already numerical:  pdist, spd, lat, lon, and eta 
3. Time is a cyclic variable. To encode this as a numerical feature, we can use a sine/cosine transformation. Suppose we have a feature of value f that ranges from 0 to N. Then, the sine and cosine transformation would be $\sin\left(2\pi \frac{f}{N}\right)$ and $\cos\left(2\pi \frac{f}{N}\right)$. For example, the sine transformation of 6 hours would be $\sin\left(2\pi \frac{6}{24}\right)$, since there are 24 hours in a cycle. You should create sine/cosine features for the following:
    * day of week (cycles every week, 0=Monday)
    * hour of day (cycles every 24 hours, 0=midnight)
    * time of day represented by total number of minutes elapsed in the day (cycles every 60*24 minutes, 0=midnight).
4. Heading is also a cyclic variable, as it is the ordinal direction in degrees (so cycles every 360 degrees). 
4. Buses run on different schedules on the weekday as opposed to the weekend. Create a binary indicator feature `weekday` that is 1 if the day is a weekday, and 0 otherwise. 
5. Route and destination are both categorical variables. We can encode these as indicator vectors, where each column represents a possible category and a 1 in the column indicates that the row belongs to that category. This is also known as a one hot encoding. Make a set of indicator features for the route, and another set of indicator features for the destination. 
6. The names of your indicator columns for your categorical variables should be exactly the value of the categorical variable. The pandas function `pd.DataFrame.get_dummies` will be useful. 

```
def create_features(vdf):
    """ Given a dataframe of labeled and truncated bus data, generate features for linear regression. 
    
        Args:
            df (dataframe) : dataframe of bus data with the eta column and truncated rows
        Return: 
            (dataframe) : dataframe of features for each example
        """
    # No change: pdist, spd, lat, lon, eta
    # Dummy: rt des
    # Drop: vid hdg pid tablockid tatripid
    import math
    cos_hdg = []
    sin_hdg = []
    cos_day_of_week = []
    sin_day_of_week = []
    cos_hour_of_day = []
    sin_hour_of_day = []
    cos_time_of_day = []
    sin_time_of_day = []
    weekday = []
    result = vdf[['pdist','spd','lat','lon','eta']].copy()
    result.insert(0, 'bias', 1.0)

    for index, row in vdf.iterrows():
        time = row['tmstmp']
        hdg_parameter = row['hdg'] / float(360) * 2 * math.pi
        day_of_week_parameter = time.weekday() / float(7) * 2 * math.pi
        hour_of_day_parameter = time.hour / float(24) * 2 * math.pi
        time_of_day_parameter = (time.hour * 60 + time.minute) / float(1440) * 2 * math.pi
        cos_hdg.append(math.cos(hdg_parameter))
        sin_hdg.append(math.sin(hdg_parameter))
        cos_day_of_week.append(math.cos(day_of_week_parameter))
        sin_day_of_week.append(math.sin(day_of_week_parameter))
        cos_hour_of_day.append(math.cos(hour_of_day_parameter))
        sin_hour_of_day.append(math.sin(hour_of_day_parameter))
        cos_time_of_day.append(math.cos(time_of_day_parameter))
        sin_time_of_day.append(math.sin(time_of_day_parameter))
        this_weekday = 1
        if time.weekday() == 5 or time.weekday() == 6:
            this_weekday = 0
        weekday.append(this_weekday)
    
    result['sin_hdg'] = pd.Series(sin_hdg, index = result.index)
    result['cos_hdg'] = pd.Series(cos_hdg, index = result.index)
    result['sin_day_of_week'] = pd.Series(sin_day_of_week, index = result.index)
    result['cos_day_of_week'] = pd.Series(cos_day_of_week, index = result.index)
    result['sin_hour_of_day'] = pd.Series(sin_hour_of_day, index = result.index)
    result['cos_hour_of_day'] = pd.Series(cos_hour_of_day, index = result.index)
    result['sin_time_of_day'] = pd.Series(sin_time_of_day, index = result.index)
    result['cos_time_of_day'] = pd.Series(cos_time_of_day, index = result.index)
    result['weekday'] = pd.Series(weekday, index = result.index)
    
    df_des = pd.get_dummies(vdf['des'])
    result = result.join(df_des)
    df_rt = pd.get_dummies(vdf['rt'])
    result = result.join(df_rt)
    return result

vdf_features = create_features(labeled_vdf)
print vdf_features.columns
print vdf_features.head()
# print vdf_features.isnull().values.any()
```

Our implementation has the following output. Verify that your code has the following columns (order doesn't matter): 
```python
>>> vdf_features.columns
Index([             u'bias',             u'pdist',               u'spd',
                     u'lat',               u'lon',               u'eta',
                 u'sin_hdg',           u'cos_hdg',   u'sin_day_of_week',
         u'cos_day_of_week',   u'sin_hour_of_day',   u'cos_hour_of_day',
         u'sin_time_of_day',   u'cos_time_of_day',           u'weekday',
               u'Braddock ',          u'Downtown',   u'Greenfield Only',
             u'McKeesport ', u'Murray-Waterfront',         u'Swissvale',
                     u'61A',               u'61B',               u'61C',
                     u'61D'],
      dtype='object')
   bias  pdist  spd        lat        lon  eta   sin_hdg   cos_hdg  \
0   1.0   1106    0  40.439504 -79.996981   16  0.913545 -0.406737   
1   1.0   1106    0  40.439504 -79.996981   15  0.913545 -0.406737   
2   1.0   1778    8  40.438842 -79.994733   14  0.829038 -0.559193   
3   1.0   2934    7  40.437938 -79.991213   13  0.997564 -0.069756   
4   1.0   2934    7  40.437938 -79.991213   13  0.997564 -0.069756   

   sin_day_of_week  cos_day_of_week ...   Braddock   Downtown  \
0         0.433884        -0.900969 ...         0.0       0.0   
1         0.433884        -0.900969 ...         0.0       0.0   
2         0.433884        -0.900969 ...         0.0       0.0   
3         0.433884        -0.900969 ...         0.0       0.0   
4         0.433884        -0.900969 ...         0.0       0.0   

   Greenfield Only  McKeesport   Murray-Waterfront  Swissvale  61A  61B  61C  \
0              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
1              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
2              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
3              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
4              0.0          0.0                0.0        1.0  1.0  0.0  0.0   

   61D  
0  0.0  
1  0.0  
2  0.0  
3  0.0  
4  0.0  

[5 rows x 25 columns]
```

## Q3 Linear Regression using Ordinary Least Squares [10 + 4pts]
Now you will finally implement a linear regression. As a reminder, linear regression models the data as

$$\mathbf y = \mathbf X\mathbf \beta + \mathbf \epsilon$$

where $\mathbf y$ is a vector of outputs, $\mathbf X$ is also known as the design matrix, $\mathbf \beta$ is a vector of parameters, and $\mathbf \epsilon$ is noise. We will be estimating $\mathbf \beta$ using Ordinary Least Squares, and we recommending following the matrix notation for this problem (https://en.wikipedia.org/wiki/Ordinary_least_squares). 

### Specification
1. We use the numpy term array-like to refer to array like types that numpy can operate on (like Pandas DataFrames). 
1. Regress the output (eta) on all other features
2. Return the predicted output for the inputs in X_test
3. Calculating the inverse $(X^TX)^{-1}$ is unstable and prone to numerical inaccuracies. Furthermore, the assumptions of Ordinary Least Squares require it to be positive definite and invertible, which may not be true if you have redundant features. Thus, you should instead use $(X^TX + \lambda*I)^{-1}$ for identity matrix $I$ and $\lambda = 10^{-4}$, which for now acts as a numerical "hack" to ensure this is always invertible. Furthermore, instead of computing the direct inverse, you should utilize the Cholesky decomposition which is much more stable when solving linear systems. 
```
class LR_model():
    """ Perform linear regression and predict the output on unseen examples. 
        Attributes: 
            beta (array_like) : vector containing parameters for the features """
    
    def __init__(self, X, y):
        """ Initialize the linear regression model by computing the estimate of the weights parameter
            Args: 
                X (array-like) : feature matrix of training data where each row corresponds to an example
                y (array like) : vector of training data outputs 
            """
#         theta1 = np.linalg.solve(X.T.dot(X), X.T.dot(y))
#         theta2 = -theta1 * np.mean(X) + np.mean(y)
        theta1 = np.linalg.solve(X.T.dot(X) + 1e-4*np.eye(X.shape[1]), X.T.dot(y)) 
        self.beta = theta1
        
        self.beta = np.zeros(X.shape[1])
        temp = np.add(X.T.dot(X), np.identity(X.shape[1]) * le-4)
        c_and_lower = la.cho_factor(temp, lower=True)
        self.beta = la.cho_solve(c_and_lower, X.T.dot(y)
        
    def predict(self, X_p): 
        """ Predict the output of X_p using this linear model. 
            Args: 
                X_p (array_like) feature matrix of predictive data where each row corresponds to an example
            Return: 
                (array_like) vector of predicted outputs for the X_p
            """
        return X_p.dot(self.beta.T)
            
        
pass_X = vdf_features.copy().drop("eta", 1)
pass_y = vdf_features.copy()["eta"]
lr_model = LR_model(pass_X, pass_y)        
```
[ -6.03556546e+03  -8.86229599e-04  -5.74005961e-02   1.29822894e+02
  -4.15270738e+01   5.20820956e-01  -4.66720834e-01   1.15809425e+00
  -7.97829204e-01   7.22449126e+00  -7.99358835e+00  -6.85149145e+00
   7.30701042e+00   4.36039537e-01  -1.00928692e+03  -9.84913932e+02
  -1.01548581e+03  -1.02542457e+03  -1.00497366e+03  -9.95479959e+02
  -1.51847098e+03  -1.50968589e+03  -1.49329158e+03  -1.51411659e+03]
bias                 6051.276738
pdist                  30.575902
spd                    16.787689
lat                 -5232.275571
lon                 -3303.344288
sin_hdg                15.687598
cos_hdg                15.823104
sin_day_of_week        15.688213
cos_day_of_week        15.537868
sin_hour_of_day        16.419188
cos_hour_of_day        12.933400
sin_time_of_day        14.735336
cos_time_of_day        18.143075
weekday                15.359417
Braddock              103.908829
Downtown              662.386089
Greenfield Only        20.784569
McKeesport            104.998243
Murray-Waterfront      98.701681
Swissvale              96.728622
61A                   324.210526
61B                   395.644458
61C                   462.271798
61D                   388.376932
dtype: float64

