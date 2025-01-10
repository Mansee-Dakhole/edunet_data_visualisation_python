#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[5]:


df=pd.read_csv("C:/Users/dakho/Downloads/agricultural_yield.csv")
df.head()


# In[7]:


X=df[[	'Soil_Quality',	'Seed_Variety',	'Fertilizer_Amount_kg_per_hectare','Sunny_Days','Rainfall_mm',	'Irrigation_Schedule']]	
y=df['Yield_kg_per_hectare']


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[11]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)


# In[13]:


model=Sequential([
                Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
                 Dense(32,activation='relu'),
                 Dense(1)])


# In[15]:


model.compile(optimizer='adam',
             loss='MSE',
             metrics=['MAE'])


# In[25]:


model.summary()


# In[28]:


history=model.fit(X_train,y_train,epochs=100,batch_size=32,validation_split=0.2)


# In[ ]:





# In[21]:


test_loss,test_mae=model.evaluate(X_test,y_test,verbose=1)
print(f"The Mean Absolute Error: {test_mae:2f}")


# In[22]:


plt.plot(history.history ['loss'], label='Training Less')
plt.plot(history.history['val_loss'], label=' Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Training Performance')
plt.legend()
plt.show()


# In[34]:


# Distribution plots
import seaborn as sns
sns.distplot(y_test, label='Actual Yield')
sns.distplot(y_pred, label='Predicted Yield')
plt.xlabel("Yield")
plt.ylabel("Density")
plt.title("Distribution of Actual and Predicted Yield")
plt.legend()
plt.show()


# In[36]:


predictions = model.predict(X_test)
predictions


# In[38]:


plt.scatter(y_test,predictions)
plt.xlabel('True Energy Consumption')
plt.ylabel('predicted Energy Consumption')
plt.title('prediction vs true values')
plt.show()

