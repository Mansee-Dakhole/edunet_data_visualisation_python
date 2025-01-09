#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
data = pd.read_csv("C:/Users/dakho/Downloads/green_tech_data.csv")
print(data)


# In[73]:


import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[27]:


data.head()


# In[75]:


missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values)


# In[77]:


# Count of each class
class_counts = data['sustainability'].value_counts()
print("Class Distribution:\n", class_counts)

# Plot class distribution
import matplotlib.pyplot as plt

class_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Sustainability')
plt.ylabel('Frequency')
plt.show()

# Check balance threshold
imbalance_ratio = class_counts.min() / class_counts.max()
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

if imbalance_ratio < 0.5:
    print("Data is imbalanced.")
else:
    print("Data is balanced.")



# In[79]:


# Count of each class
class_counts = data['sustainability'].value_counts()
print("Class Distribution:\n", class_counts)

# Plot class distribution
import matplotlib.pyplot as plt

class_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Sustainability')
plt.ylabel('Frequency')
plt.show()

# Check balance threshold
imbalance_ratio = class_counts.min() / class_counts.max()
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

if imbalance_ratio < 0.5:
    print("Data is imbalanced.")
else:
    print("Data is balanced.")



# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()


# In[83]:


# Features (independent variable) and target (dependent variable)
# Independent variable: 'carbon_emissions'
X = data['carbon_emissions']

# Dependent variable: 'energy_output'
Y = data['energy_output']

print("Independent Variable (X):\n", X.head())
print("Dependent Variable (Y):\n", Y.head())


# In[85]:


# Features (independent variable) and target (dependent variable)
# Independent variable: 'carbon_emissions'
X = data['carbon_emissions']

# Dependent variable: 'energy_output'
Y = data['energy_output']

print("Independent Variable (X):\n", X.head())
print("Dependent Variable (Y):\n", Y.head())


# In[87]:


# Features (independent variable) and target (dependent variable)
# Independent variable: 'carbon_emissions'
X = data['carbon_emissions']

# Dependent variable: 'energy_output'
Y = data['energy_output']

print("Independent Variable (X):\n", X.head())
print("Dependent Variable (Y):\n", Y.head())


# In[89]:


# Features (independent variables) and target (dependent variable)
# Independent variables: carbon_emissions, renewability_index, cost_efficiency
X = data[['carbon_emissions', 'renewability_index', 'cost_efficiency']]

# Dependent variable: energy_output
Y = data['energy_output']

print("Independent Variables (X):\n", X.head())
print("Dependent Variable (Y):\n", Y.head())


# In[91]:


print(df.isnull().sum())


# In[93]:


print(df.isnull().sum())


# In[119]:


# Example feature selection
X=data[ ['carbon_emissions', 'energy_output', 'renewability_index', 'cost_efficiency']]
y=data['sustainability']


# In[121]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[123]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[125]:


#predication
y_pred=model.predict(X_test)


# In[127]:


#accuracy
from sklearn.metrics import accuracy_score
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


# In[129]:


#confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(y_test,y_pred)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Not Sustainable', 'Sustainable'],
           yticklabels=['Not Sustainable', 'Sustainable'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()


# In[131]:


#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred,target_names=['Not Sustainable', 'Sustainable']))


# In[133]:


# Feauture importance
coefficients = pd.DataFrame(model.coef_.T,index=X.columns,columns=['Coefficient'])
print(coefficients)


# In[147]:


import joblib
joblib.dump(model, 'lrmodel_sustainability.pkl')  # Save model with .pkl extension


# In[151]:


model = joblib.load('lrmodel_sustainability.pkl')  # Load model with .pkl extension


# In[153]:


import numpy as np
# Assuming 'model' is your trained model
new_data = np.array([[22.49,25, 0.85, 0.72,]])  # Example values for carbon_emissions, renewability_index, cost_efficiency
predictions = model.predict(new_data)
print("Output:",predictions)
if predictions==1:
    print("Sustainable")
else:
    print("Non-Sustainable")


# In[ ]:




