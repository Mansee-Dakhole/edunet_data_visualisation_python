#!/usr/bin/env python
# coding: utf-8

# In[20]:


#pair plot 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
df = pd.read_csv("C:/Users/dakho/OneDrive/Desktop/IRIS.csv")
print(df)
df = pd.DataFrame(df)
sns.pairplot(df,hue='species',palette="husl",height=2)
plt.show()


# In[22]:


#heatmap
# Filter numeric columns
numeric_df = df.select_dtypes(include='number')

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Set the title
plt.title('Correlation between iris-setosa')

# Show the plot
plt.show()


# In[26]:


from sklearn.preprocessing import LabelEncoder

# Assign numerical values to the species column
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['species'])

# Filter numeric columns
numeric_df = df.select_dtypes(include='number')

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Set the title
plt.title('Correlation between iris-species')

# Show the plot
plt.show()


# In[28]:


from sklearn.preprocessing import LabelEncoder

# Assign numerical values to the species column
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['species'])

# Filter numeric columns
numeric_df = df.select_dtypes(include='number')

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Set the title
plt.title('Correlation between iris-species')

# Show the plot
plt.show()


# In[13]:


#box plot 
sns.boxplot(data=df)

plt.title('Distribution of iris-setosa')
plt.show()


# In[15]:


#violin plot 
sns.violinplot(data=df)

plt.title('violin plot of ')
plt.show()


# In[30]:


# Scatter plot
plt.figure(figsize=(10, 7))
sns.scatterplot(x='petal_length', y='petal_width', hue='Species', data=df)

# Set plot title and labels
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Show the plot
plt.show()

