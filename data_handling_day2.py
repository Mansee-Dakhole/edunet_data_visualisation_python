#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd 
import numpy as np

# Create the DataFrame
data = {
    "Energy Source": ["Solar", "Wind", "Hydropower", "Geothermal", "Biomass", "Nuclear"],
    "Energy Consumption (MWh)": [1200, np.nan, 2900, np.nan, 2500, 3200],
    "Cost (Million $)": [200, 400, np.nan, 150, 250, np.nan]
}

df = pd.DataFrame(data)
energy_df=pd.DataFrame(data)

# remove rws with any missing values 
cleaned_df = energy_df.dropna()
print("\nData after removing rows with missing values:")
cleaned_df.head()


# In[17]:


# relace the missing value with mean 
ec_mean = energy_df["Enery consumption(MWH)"].mean()
print("Mean of Enery Consumption (MWH)",ec_mean)
cost_mean = energy_df["Cost (Million $)"].mean()
print("Mean of Cost (Million $)", cost_mean)


# In[19]:


cleaned_df['Cost (Million $)'].mean()


# In[21]:


cleaned_df['Energy Consumption (MWh)'].mean()


# In[23]:


cleaned_df['Cost (Million $)'].mean()


# In[25]:


energy_df['Cost (Million $)'].fillna(cost_mean,inplace=True)
energy_df['Energy Consumption (MWh)'].fillna(ec_mean,inplace=True)
energy_df.head()


# In[ ]:





# In[27]:


#forward fill missing fill 
forward_filled_df = energy_df.fillna(method="ffill")
print("\nData Before foreward filling:")
print(energy_df)
print("\nData After forward filling:")
forward_filled_df.head()


# In[29]:


#create a flag column indicacating missing values in enery consumption (mwh)
# Create a flag column indicating missing values in 'Energy Consumption (MWh)'
energy_df["Missing Consumption"] = energy_df["Energy Consumption (MWh)"].isna().astype(int)
energy_df["Missing Cost"] = energy_df["Cost (Million $)"].isna().astype(int)

print("\nData with Missing Values Flagged:")
energy_df.head()


# In[ ]:


#preprossesing data 
#mean max scalar(x`)=x - min(x)/ max(x)-min(x)

xmin_ec = np.min(energy_df[["Energy Consumption ")


# In[33]:


from sklearn.preprocessing import MinMaxScaler  # Correct spelling of MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the specified columns using Min-Max Scaling
energy_df[["Energy Consumption (MWh)", "Cost (Million $)"]] = scaler.fit_transform(
    energy_df[["Energy Consumption (MWh)", "Cost (Million $)"]]
)

print("\nData After Normalization (Min-Max Scaling):")
print(energy_df)


# In[39]:


#data scaling (z score scaling )
# formula - Z= σ/X−μ

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply Z-score scaling to the specified columns
energy_df[["Energy Consumption (MWh)", "Cost (Million $)"]] = scaler.fit_transform(
    energy_df[["Energy Consumption (MWh)", "Cost (Million $)"]]
)

print("\nData After Z-Score Scaling:")
print(energy_df)


​
 


# In[55]:


# enoding categorical variable energy source colume
# one-hot encode the 'Energynsource ' column 
import pandas as pd
import numpy as np

# Original data
data = {
    "Energy Source": ["Solar", "Wind", "Hydropower", "Geothermal", "Biomass", "Nuclear"],
    "Energy Consumption (MWh)": [1200, np.nan, 2900, np.nan, 2500, 3200],
    "Cost (Million $)": [200, 400, np.nan, 150, 250, np.nan]
}

# Convert the data dictionary to a DataFrame
energy_df = pd.DataFrame(data)

# One-hot encode the 'Energy Source' column
energy_encoded_df = pd.get_dummies(energy_df, columns=["Energy Source"], prefix="Source")

# Display the resulting DataFrame
print("One-Hot Encoded DataFrame:")
print(energy_encoded_df)

