#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Single linear Reg.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#Load dataset
mall_data = pd.read_csv('F:/Mall_Customers.csv')


# In[17]:


#Select features and target 
X = mall_data[['Age']]  #Predictor:Age
y = mall_data['Spending_Score']  #Target:Spending_Score

#Split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Apply Linear Reg.
model = LinearRegression()
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

#Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r2)
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)


# In[19]:


#Graph
plt.figure(figsize=(10, 6))

#Scatter plot 
sns.scatterplot(x=X_test['Age'], y=y_test, color='black', label='Data')

#Regression line
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

plt.title('Linear Regression: Age vs Spending Score', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Spending Score', fontsize=14)
plt.legend(fontsize=12)

plt.show()


# In[20]:


#Multiple Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Load dataset
mall_data = pd.read_csv('F:/Mall_Customers.csv')

#Preprocessing: Convert Gender to numeric values
mall_data['Gender'] = mall_data['Gender'].map({'Male': 0, 'Female': 1})

#predictors (Age, Annual Income) and target (Spending_Score)
X = mall_data[['Age', 'Annual_Income_(k$)']]  #Predictors
y = mall_data['Spending_Score']  #Target


# In[21]:


#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)


# In[22]:


#evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
coefficients = model.coef_
intercept = model.intercept_

print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r2)
print("Coefficients (Slopes):", coefficients)
print("Intercept:", intercept)

#visualize
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.title('Actual vs Predicted Spending Scores', fontsize=16)
plt.xlabel('Actual Spending Score', fontsize=14)
plt.ylabel('Predicted Spending Score', fontsize=14)
plt.grid()
plt.show()


# In[ ]:




