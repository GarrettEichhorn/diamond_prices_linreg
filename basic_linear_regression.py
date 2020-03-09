# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import joblib

# Importing the dataset
file_path = os.path.join("data", "diamonds.csv")
dataset = pd.read_csv(file_path)

ind_variable = dataset['carat'].values.reshape(-1,1)
dep_variable = dataset['price'].values.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(ind_variable, dep_variable, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Compare model performance!
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# Saving model to disk
joblib.dump(regressor, 'models/linear_regression.pkl')

