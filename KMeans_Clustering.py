#!/usr/bin/env python
# coding: utf-8

# <span style="color:black;font-size: 30px; font-family: Arial; font-weight: bold;">EDUNET FOUNDATION-Classroom Exercise Notebook</span>

# Classroom Exercise : Implementing K-means Algorithm

# # Problem Statement: 
# 
# A retail store wants to get insights about its customers. And then build a system that can cluster customers into different groups.

# # Steps:

# ### Import Libraries

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = '1'

#This can prevent multi-threading conflicts, reduce CPU load, or help in debugging.
#Some libraries or operations (e.g., NumPy operations or machine learning libraries) use OpenMP to parallelize tasks across multiple threads by default. This can sometimes lead to issues, such as excessive CPU usage


# In[3]:


import pandas as pd


# In[5]:


import matplotlib.pyplot as plt


# In[7]:


import seaborn as sns


# In[8]:


from sklearn.cluster import KMeans


# ### Loading Data

# In[16]:


df=pd.read_csv("C:/Users/dakho/Downloads/Mall_Customers.csv")


# The data includes the following features:
# 
# 1. Customer ID
# 2. Customer Gender
# 3. Customer Age
# 4. Annual Income of the customer (in Thousand Dollars)
# 5. Spending score of the customer (based on customer behaviour and spending nature)

# In[18]:


df.head()


# ### Data Exploration

# In[17]:


### Check Null Values


# In[20]:


df.isnull().sum()


# In[19]:


### Observation: There is no missing values.


# In[20]:


### Visual and Statistical Understanding of data


# In[22]:


df.columns


# In[24]:


plt.scatter(df['Age'],df['Spending Score (1-100)'])
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.show()


# In[23]:


### Observation: It seems to purpose two types of Customer


# In[26]:


plt.scatter(df["Age"],df["Annual Income (k$)"])
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)") 
plt.show()


# In[28]:


plt.scatter(df["Spending Score (1-100)"],df["Annual Income (k$)"])
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)") 
plt.show()


# In[25]:


### Observation: No Group


# In[30]:


sns.scatterplot(x=df["Spending Score (1-100)"], y=df["Annual Income (k$)"], hue=df['Gender'])
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.show()


# In[27]:


### It seems to purpose five Groups


# ### Choose Relevant Columns

# All the columns are  not relevant for the clustering. In this example, we will use the numerical ones: Age, Annual Income, and Spending Score

# In[32]:


relevant_cols = ["Age", "Annual Income (k$)", 
                 "Spending Score (1-100)"]

customer_df = df[relevant_cols]


# In[34]:


customer_df


# ### Data Transformation

# Kmeans is sensitive to the measurement units and scales of the data. It is better to standardize the data first to tackle this issue

# The standardization substracts the mean of any feature from the actual values of that feature and divides the feature’s standard deviation.

# In[36]:


from sklearn.preprocessing import StandardScaler


# In[38]:


scaler = StandardScaler()


# In[40]:


scaler.fit(customer_df)


# In[42]:


scaled_data = scaler.transform(customer_df)


# If you're using a StandardScaler, it scales each feature to have zero mean and unit variance.
# 
# If you're using a MinMaxScaler, it scales each feature to be in the range [0, 1].

# In[44]:


scaled_data


# ### Determine the best number of cluster

# A clustering model will not be relevant if we fail to identify the correct number of clusters to consider. Multiple techniques exist in the literature. We are going to consider the Elbow method, which is a heuristic method, and one of the widely used to find the optimal number of clusters.

# In[46]:


def find_best_clusters(df, maximum_K):
    clusters_centers = []
    k_values = []
    for k in range(2, maximum_K):
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)

        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)

    return clusters_centers, k_values

#measure of how tightly grouped the data points are within their clusters using elbow method. The inertia decreases as the number of clusters increases, but at some point, adding more clusters gives diminishing returns


# In[48]:


clusters_centers, k_values = find_best_clusters(scaled_data, 12)


# In[52]:


def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()


# In[54]:


generate_elbow_plot(clusters_centers, k_values)


# From the plot, we notice that the cluster inertia decreases as we increase the number of clusters. Also the drop the inertia is minimal after K=5 hence 5 can be considered as the optimal number of clusters.

# ### Create the final KMeans model

# In[56]:


kmeans_model = KMeans(n_clusters = 5)


# In[58]:


kmeans_model.fit(scaled_data)


# In[51]:


### We can access the cluster to which each data point belongs by using the .labels_ attribute.


# In[60]:


df["clusters"] = kmeans_model.labels_


# In[62]:


df


# ### Visualize the clusters

# In[66]:


plt.scatter(df["Spending Score (1-100)"], 
            df["Annual Income (k$)"], 
            c = df["clusters"]
            )


# The KMeans clustering seems to generate a pretty good result, and the five clusters are well separated from each other, even though there is a slight overlap between the purple and the yellow clusters.

# - Customers on the top left have a low spending score and a high annual income. A good marketing strategy could be implemented to target those customers so that they can spend more.
# - On the other hand, customers on the bottom left have a low annual income and also spends less, which makes sense, because they are trying to adjust their spending habit to their budget.
# - The top right customers are similar to the bottom left, the difference is that they have enough budget to spend.
# - Finally, the yellow group of customers spends beyond their budget.
