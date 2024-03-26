import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("student_spending.csv")
df = df.drop("Unnamed: 0", axis =1)

X = df.drop(columns=['preferred_payment_method','gender','year_in_school','major'])
y_value = df['preferred_payment_method']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_value)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# # split the wave dataset into a training and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size =0.3)

# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)

print("Training set R^2: {:.2f}".format(reg.score(X_train, y_train)))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


slr = linear_model.LinearRegression() #create an linear regression model objective 

slr.fit(X_train,y_train) # estimate the patameters
print('beta',slr.coef_)
print('alpha',slr.intercept_)


y_predict = slr.predict(X_test) # predict the Y based on the model
mean_squared_error = mean_squared_error(y_test,y_predict) # calculate mean square error
r2_score = r2_score(y_test,y_predict) #calculate r square

print ('mean square error:',mean_squared_error )
print ('r square:',r2_score )


