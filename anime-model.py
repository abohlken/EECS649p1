from sklearn import *
from pandas import *
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import *
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pandas.read_csv('animedata/UserList.csv')

data.drop(columns=['username', 'user_id','gender', 'location', 'birth_date', 'access_rank', 'join_date', 'last_online'], inplace=True)
data.dropna(inplace=True)

data_y = data.iloc[:,7]
data_x = data.iloc[:, [True, True, True, True, True, True, True, False, True]]
print(data.columns)

selector = VarianceThreshold()
selector.fit_transform(data_x)

X2 = sm.add_constant(data_x)
est = sm.OLS(data_y, X2)
est2 = est.fit()
print(est2.summary())

data_x_train = data_x[:-30000]
data_x_test = data_x[-30000:]

data_y_train = data_y[:-30000]
data_y_test = data_y[-30000:]

regr = linear_model.LinearRegression()

regr.fit(data_x_train, data_y_train)

predictions = regr.predict(data_x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(data_y_test, predictions))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(data_y_test, predictions))