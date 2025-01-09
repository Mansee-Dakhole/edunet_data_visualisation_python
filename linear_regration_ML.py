#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
df = pd.read_csv("C:/Users/dakho/Downloads/appliance_energy.csv")
print(df)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[5]:


#scatter plot
plt.scatter(df['Temperature (°C)'], df['Energy Consumption (kWh)'])
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.show()


# In[7]:


df.describe()


# In[9]:


#check for the missing values
print(df.isnull().sum())
#df = df.dropna()


# In[11]:


# Feature (independent variable) and target (dependent variable)
# Independent variable: Temperature
X = df[['Temperature (°C)']]  # Correct column name for independent variable
y = df['Energy Consumption (kWh)']  # Correct column name for dependent variable

# Display shapes of X and y
print("Shape of X (independent variable):", X.shape)
print("Shape of y (dependent variable):", y.shape)


# In[13]:


#split the data into training and testing sets 
# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the splits
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[15]:


X_test.head()


# In[17]:


y_test.head()


# In[19]:


# create a linear Regration model 
model = LinearRegression()
model.fit(X_train,y_train)


# In[21]:


print("Slope: ",model.coef_ )
print("y-intercept: ",model.intercept_)


# In[23]:


#y=mx+c
print(model.coef_*28.25 + model.intercept_)


# In[25]:


#make prediction on the test set
y_pred = model.predict(X_test)
y_pred


# In[27]:


#calculate mean square error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean square Error: {mse} ")

#calculate R-squared value
r2 = r2_score(y_test,y_pred)
print(f"R-Squared: {r2}")


# In[29]:


#plot the test data and regression line 
plt.scatter(X_test,y_test,color='blue',label="Test Data")
plt.plot(X_test,y_pred,color='red',label="Regression Line")
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Temperature vs Energy Consumption')


# In[31]:


import joblib
#save the model to a file 
joblib.dump(model,'appliance_enery_model.pkl')


# In[33]:


temp=np.array([[22.49]])


# In[35]:


prediction =  model.predict(temp)
print("prediction energy consumption:",prediction)


# In[37]:


temp=np.array([22.49])
prediction=model.predict(temp.reshape(1,-1))
print(prediction)


# In[60]:


#energy pred
prediction = model.predict(temp)
print("Pred Energy consumption:",prediction)


# In[43]:


#to use the pkl file for predication 
loaded_model=joblib.load('appliance_energy_model.pkl')
result=loaded_model.predict([[28.25]])
print(result)

