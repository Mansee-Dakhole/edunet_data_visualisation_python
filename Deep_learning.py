#!/usr/bin/env python
# coding: utf-8

# In[48]:


pip install tensorflow


# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the CSV file directly from its file path
file_path = "C:/Users/dakho/Downloads/green_tech_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())


# In[82]:


x = df[['carbon_emissions', 'renewability_index','cost_efficiency']].values
y = df['sustainability'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[74]:


model=Sequential([
    Dense(64,activation='relu',input_shape=(3,)),
    Dense(12,activation='relu'),
    Dense(1,activation='sigmoid')
])


# In[84]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])


# In[86]:


model.summary()


# In[88]:


#train
model.fit(x_train, y_train,epochs=30,batch_size = 16)


# In[90]:


y_pred = model.predict(x_test)
y_pred


# In[ ]:




