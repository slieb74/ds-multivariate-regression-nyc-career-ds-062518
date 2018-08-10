
# Here's some data!

### 1. Import the data. It's stored in a file called 'movie_data_detailed.xlsx'.


```python
df = None #Your code here
```

### 2. Fill all the null values with zero.


```python
#Your code here
```

### 3. Normalize the data so that all features have a minimum of zero and a maximum of one.


```python
#Your code here
```

### 4. Define 4 variables: X_train, Y_train, X_test, Y_test using a 80-20 split for train and test data. X should be a matrix of data features predicting y, Domestic Gross Sales.  Use random_state=42 for consistency.


```python
from sklearn.model_selection import train_test_split
```


```python
X = None
y = None
```


```python
#Your code here
```

### 5. Import import sklearn.linear_model.LinearRegression
Create an instance of the LinearRegression class.
Then use the fit method to train a model according to the data.


```python
import sklearn.linear_model as linreg
```


```python
#Create Instance of LinearRegression (Ordinary Least Squares Regressor)
```


```python
#Fit the model to the train set
```

### 6. Scatter Plot <a id="scatter"></a>  
Create a Scatter Plot of the budget and  Domestic Gross (domgross) along with your model's predictions.


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
#Scatter Plot
```

### 7. Calculate the RSS for both the train and test sets.
Define a function called rss(y,y_hat). Call it on the train and test sets.


```python
def rss(y, y_hat):
    pass
```


```python
# print('RSS Training: {}'.format())
# print('RSS Test: {}'.format())
```
