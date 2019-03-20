from sklearn import *
from pandas import *
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import *
from sklearn.preprocessing import *
import statsmodels.api as sm
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.neighbors import KNeighborsClassifier

#read in data
data = pandas.read_csv('openpowerlifting.csv')

### clean up data
data.drop(columns=['MeetID', 'Name', 'Division', 'Squat4Kg', 'BestSquatKg', 'BestBenchKg', 'Deadlift4Kg', 'BestDeadliftKg', 'Place', 'WeightClassKg'], inplace=True)

data['SexNum'] = data['Sex'].transform(lambda x : 0 if x == 'M' else 1)
data.drop(columns=['Sex'], inplace=True)

data['EquipNum'] = data['Equipment'].transform(lambda x : 0 if x == 'Raw' else (1 if x == 'Single-ply' else (2 if x == 'Multi-ply' else (3 if x== 'Wraps' else 4))))
data.drop(columns=['Equipment'], inplace=True)

data.fillna(0, inplace=True)

###find stats about model
data_y = data.iloc[:,4]
data_x = data.iloc[:, [False, True, False, True, False, True, False]]
print(data.columns)

X2 = sm.add_constant(data_x)
est = sm.OLS(data_y, X2)
est2 = est.fit()
print(est2.summary())

### exhaustive search for best model
# knn = KNeighborsClassifier()

# efs1 = EFS(knn, 
           # min_features=1,
           # max_features=6,
           # scoring='accuracy',
           # print_progress=True)

# efs1 = efs1.fit(data_x, data_y.astype('int'))

# print('Best accuracy score: %.2f' % efs1.best_score_)
# print('Best subset (indices):', efs1.best_idx_)
# print('Best subset (corresponding names):', efs1.best_feature_names_)

###split into training and testing set
data_x_train = data_x[:-30000]
data_x_test = data_x[-30000:]

data_y_train = data_y[:-30000]
data_y_test = data_y[-30000:]

###linear regression
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
# Coefficients:
 # [-1.13882495  0.65056555 76.60414171]
# Mean squared error: 1255.54
# Variance score: 0.94

regrLas = linear_model.Lasso(alpha=0.1)
regrLas.fit(data_x_train, data_y_train.astype('int'))
predictionsLas = regrLas.predict(data_x_test)

# The coefficients
print('Coefficients: \n', regrLas.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(data_y_test, predictionsLas))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(data_y_test, predictionsLas))
# Coefficients:
 # [-1.14220886  0.6499499  75.82868729]
# Mean squared error: 1254.26
# Variance score: 0.94