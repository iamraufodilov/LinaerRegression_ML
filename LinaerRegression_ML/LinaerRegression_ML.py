# import libraries
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# load data
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]

# splitting data
X_train = X[:-30]
X_test = X[-30:]
y_train = diabetes.target[:-30]
y_test = diabetes.target[-30:]

# create model
LReg = linear_model.LinearRegression()

# train model
LReg.fit(X_train, y_train)

# make predict
y_pred = LReg.predict(X_test)

# get report
print('Coefficients: \n', LReg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
