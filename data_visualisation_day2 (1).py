#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt 

#sample data for energy  consumption over 6 month (in mwh)
import matplotlib.pyplot as plt

#Sample data for energy consumption over 6 months (in Mich)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
energy_consumption= [1200, 1300, 1100, 1500, 1400, 1600]


#Create a Line plot
plt.plot(months, energy_consumption, marker='o', color='b', linestyle='--')

#Add titles and Labels
plt.title('Energy Consumption Over 6 Months')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (MWh)')
plt.show()


# In[7]:


import matplotlib.pyplot as plt
months = ['Jan','Feb','Mar','Apr','May','Jun']
energy_consumption = [1200,1300,1100,1500,1400,1600]

plt.bar(months,energy_consumption,color='b')

plt.title('Energy Consumption Over 6 Months')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (MWh)')
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Data for energy consumption over 6 months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
energy_consumption = [1200, 1300, 1100, 1500, 1400, 1600]

# Create a pie chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(
    energy_consumption, 
    labels=months, 
    autopct='%1.1f%%',  # Show percentages
    startangle=90,  # Start angle at 90 degrees
    colors=['lightblue', 'lightgreen', 'orange', 'pink', 'violet', 'yellow'],  # Custom colors
    explode=[0, 0.1, 0, 0, 0, 0]  # Highlight February slightly
)

# Add a title
plt.title('Energy Consumption Distribution Over 6 Months', fontsize=14)

# Show the plot
plt.show()


# In[11]:


import matplotlib.pyplot as plt

# Data for energy consumption over 6 months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
energy_consumption = [1200, 1300, 1100, 1500, 1400, 1600]

# Assign numeric values to months for the x-axis
month_numbers = range(1, len(months) + 1)

# Create a scatter plot
plt.scatter(month_numbers, energy_consumption, color='purple', s=100, edgecolors='black')

# Add labels and title
plt.title('Energy Consumption Over 6 Months', fontsize=14)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Energy Consumption (MWh)', fontsize=12)

# Replace numeric x-ticks with month names
plt.xticks(month_numbers, months)

# Show the plot
plt.show()


# In[13]:


# Sample data for energy consumption and carbon emissions
#A scatter plot shows the relationship and correlation between two variables using points on a 2D graph.
carbon_emissions = [400, 500, 450, 300, 350, 550] # in kg CO2

#Create a scatter plot
plt.scatter(energy_consumption, carbon_emissions, color='red')

#Add titles and Labels
plt.title('Energy Consumption vs Carbon Emissions')
plt.xlabel('Energy Consumption (MWh)')
plt.ylabel('Carbon Emissions (kg CO2)')
plt.show()


# In[15]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (e.g., energy consumption values)
data = [1200, 1300, 1100, 1500, 1400, 1600, 1800, 2000, 2200, 2300, 2500]

# Create a histogram with 5 bins
plt.hist(data, bins=5, color='b', edgecolor='black')

# Add titles and labels
plt.title('Energy Consumption Distribution')
plt.xlabel('Energy Consumption (MWh)')
plt.ylabel('Frequency')

# Show the plot
plt.show()



# In[27]:


#histogram 
#A histogram shows the distribution of a dataset by grouping values into bins to visualize frequency.
import matplotlib.pyplot as plt 
import numpy as np
data = np.random.normal(170,10,250)
plt.hist(data, bins=30);
plt.show()


# In[29]:


#A double line plot compares trends or changes between two related datasets over the same variable, like time.
import numpy as np 
import matplotlib.pyplot as plt 

#y-axis values 
y1=[2,3,4.5]

#y-axis values
y2=[1,1.5,5]

#function to plot 
plt.plot(y1)
plt.plot(y2)

plt.legend(["blue","green"],loc="lower right")

plt.show()


# In[2]:


# pair plot by using seaborn 
#A pair plot visualizes relationships, distributions, and patterns between multiple variables in a dataset.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 

data={
    "Solar":[1200,1400,1300],
    "Wind":[3400,3600,3200],
    "Hydropower":[2900,3100,2800],
    "Biomas":[2500,2700,2400]
}
df = pd.DataFrame(data)
sns.pairplot(df)
plt.show()

    


# In[4]:


# heatmap
# value should be closed to 1 
#A heatmap visualizes data intensity or relationships in a matrix format using color gradients, often for correlation or comparisons.
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidths=0.5)

plt.title('Correlation between energy sources')
plt.show()


# In[6]:


#box plot 
#A box plot visualizes data distribution, showing median, quartiles, variability, and outliers.

sns.boxplot(data=df)

plt.title('Distribution of energy Consumption by sources')
plt.show()



# In[8]:


#violin plot
# one value should be categorical and another continuous
#A violin plot shows data distribution, density, and variability, combining a box plot and a kernel density plot.

sns.violinplot(data=df)

plt.title('violin plot of enegry consumption distribution ')
plt.show()



# In[12]:


# regration 
#A reg plot visualizes the relationship between two variables and includes a regression line to show trends or correlations.
import seaborn as sns
import matplotlib.pyplot as plt
energy_values = [100,200,300,400,500]
carbon_emissions = [10,20,25,40,50]
df_reg = pd.DataFrame({
    'Energy Consumption': energy_values,
    'Carbon Emissions': carbon_emissions
})
sns.regplot(x='Energy Consumption',y='Carbon Emissions',data=df_reg)


# In[20]:


# 3d plot 
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

ax=plt.axes(projection= '3d')

z=np.linspace(0,1,1000)
x=z*np.sin(25 * z)
y=z*np.cos(25 * 5)

ax.plot3D(x,y,z,'green')
ax.set_title('3D line plot')
plt.show()

