#!/usr/bin/env python
# coding: utf-8

# In[9]:


# create a string variable 
city_name = "city B"
print(city_name)


# In[13]:


# write a code for degrees celsius to degrees Fahrenheit.

celsius = float(input("Enter temperature in Celsius: "))

fahrenheit = (celsius * 9/5) + 32

print(f"{celsius}°C is equal to {fahrenheit}°F")



# In[15]:


#create one variable for carbon foot print assign some values use the comparison operator Compare are good or not  
carbon_footprint = 4.5 

threshold = 7.0

if carbon_footprint < threshold:
    print(f"Your carbon footprint of {carbon_footprint} metric tons is good!")
else:
    print(f"Your carbon footprint of {carbon_footprint} metric tons is too high. Consider reducing it!")


# In[23]:


#use the indenty operator to find if the the given input is digit or not .

input_value = input("Enter a number: ")

if input_value.isdigit() is True:
    print(f"{input_value} is a digit.")
else:
    print(f"{input_value} is not a digit.")




# In[25]:


# the element is that element present in the list or not use membership operator 

my_list = [10, 20, 30, 40, 50]
element = 30

if element in my_list:
    print(f"{element} is present in the list.")
else:
    print(f"{element} is not present in the list.")


# In[35]:


# write a code to build the calculator 
# Main program
print("Select operation:")
print("1. Add")
print("2. Subtract")
print("3. Multiply")
print("4. Divide")

# Take input from the user for the operation
operation = input("Enter operation (1/2/3/4): ")

# Take two numbers as input
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Perform the chosen operation
if operation == '1':
    result = num1 + num2
    print(f"{num1} + {num2} = {result}")
elif operation == '2':
    result = num1 - num2
    print(f"{num1} - {num2} = {result}")
elif operation == '3':
    result = num1 * num2
    print(f"{num1} * {num2} = {result}")
elif operation == '4':
    if num2 != 0:
        result = num1 / num2
        print(f"{num1} / {num2} = {result}")
    else:
        print("Cannot divide by zero")
else:
    print("Invalid input")




# In[43]:


climate_data = [
    {"city": "City A", "temperature": 25, "carbon_footprint": 500},
    {"city": "City B", "temperature": 30, "carbon_footprint": 350},
    {"city": "City C", "temperature": 22, "carbon_footprint": 600},
    {"city": "City D", "temperature": 15, "carbon_footprint": 200},
    {"city": "City E", "temperature": 28, "carbon_footprint": 450},
]
high_temp_threshold=22
high_temp_cities=[city for city in climate_data if city["temperature"]>high_temp_threshold]
print("Cities with higher temperatures (>26 degree cel):")
for city in high_temp_cities:
  print(f"{city['city']}-{city['temperature']} degree cel")


# In[45]:


# find the avg carbon footprint using loops in the given data 
# Climate data
climate_data = [
    {"city": "City A", "temperature": 25, "carbon_footprint": 500},
    {"city": "City B", "temperature": 30, "carbon_footprint": 350},
    {"city": "City C", "temperature": 22, "carbon_footprint": 600},
    {"city": "City D", "temperature": 15, "carbon_footprint": 200},
    {"city": "City E", "temperature": 28, "carbon_footprint": 450},
]


total_carbon_footprint = 0
count = 0


for data in climate_data:
    total_carbon_footprint += data["carbon_footprint"]
    count += 1

average_carbon_footprint = total_carbon_footprint / count

print(f"The average carbon footprint is: {average_carbon_footprint}")


# In[51]:


# write a function if the city is sustenable or not 
climate_data = [
    {"city": "City A", "temperature": 25, "carbon_footprint": 500},
    {"city": "City B", "temperature": 30, "carbon_footprint": 350},
    {"city": "City C", "temperature": 22, "carbon_footprint": 600},
    {"city": "City D", "temperature": 15, "carbon_footprint": 200},
    {"city": "City E", "temperature": 28, "carbon_footprint": 450},
]

# Define sustainability criteria
for city in climate_data:
    carbon_footprint = city["carbon_footprint"]
    temperature = city["temperature"]
    city_name = city["city"]

    # Check if the city is sustainable
    if carbon_footprint < 400 and 18 <= temperature <= 30:
        print(f"{city_name} is sustainable.")
    else:
        print(f"{city_name} is not sustainable.")


# In[1]:


# to check the given year is a leap year or not use the lambda function
is_leap_year = lambda year: (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
year = int(input("Enter a year: "))
if is_leap_year(year):
    print(f"{year} is a leap year.")
else:
    print(f"{year} is not a leap year.")


# In[13]:


import pandas as pd 
renewable_sources = ["Solar", "Wind", "Hydropower", "Geothermal", "Biomass"]

data = {
    "Project": ["Solar Farm A", "Wind Turbine X", "Hydropower Y", "Solar Roof Z", "Geothermal Plant P"],
    "Technology": ["Solar", "Wind", "Hydropower", "Solar", "Geothermal"],
    "Capacity (MW)": [150, 300, 200, 50, 100],  # Megawatts
    "Cost (Million $)": [200, 400, 350, 100, 250],  # Project cost
    "Location": ["California", "Texas", "Washington", "Nevada", "Idaho"],
    "Completion Year": [2023, 2024, 2022, 2025, 2023]
}
#create the data frame 
projects_df = pd.DataFrame(data)
projects_df.head()
projects_df.tail()
print(projects_df)






# In[17]:


import pandas as pd 

renewable_sources = ["Solar", "Wind", "Hydropower", "Geothermal", "Biomass"]

data = {
    "Project": ["Solar Farm A", "Wind Turbine X", "Hydropower Y", "Solar Roof Z", "Geothermal Plant P"],
    "Technology": ["Solar", "Wind", "Hydropower", "Solar", "Geothermal"],
    "Capacity (MW)": [150, 300, 200, 50, 100],  # Megawatts
    "Cost (Million $)": [200, 400, 350, 100, 250],  # Project cost
    "Location": ["California", "Texas", "Washington", "Nevada", "Idaho"],
    "Completion Year": [2023, 2024, 2022, 2025, 2023]
}

# Create the DataFrame
projects_df = pd.DataFrame(data)

# Extract the second and third rows
result = projects_df.iloc[1:3]  # Indexes 1 and 2 are the second and third rows

print(result)


# In[19]:


# it gives the info of data
projects_df.info()


# In[23]:


# it gies the statistical values 
# if the mean value and and actual value 0 hai to variance bhi 0 hai matlab sabhi value same hai // use less data hai which we can remmove that (significent of std)
# 25% of data below 100// 50 % values belo 150// 75% values ae below 200 
projects_df.describe()


# In[25]:


# filter the project with capacity greater than 100

import pandas as pd 

# Data
data = {
    "Project": ["Solar Farm A", "Wind Turbine X", "Hydropower Y", "Solar Roof Z", "Geothermal Plant P"],
    "Technology": ["Solar", "Wind", "Hydropower", "Solar", "Geothermal"],
    "Capacity (MW)": [150, 300, 200, 50, 100],  # Megawatts
    "Cost (Million $)": [200, 400, 350, 100, 250],  # Project cost
    "Location": ["California", "Texas", "Washington", "Nevada", "Idaho"],
    "Completion Year": [2023, 2024, 2022, 2025, 2023]
}

# Create the DataFrame
projects_df = pd.DataFrame(data)

# Filter projects with capacity greater than 100
filtered_projects = projects_df[projects_df["Capacity (MW)"] > 100]

print(filtered_projects)


# In[27]:


# add a new colum to calculate the cost per mw why it is imp?
import pandas as pd 

# Data
data = {
    "Project": ["Solar Farm A", "Wind Turbine X", "Hydropower Y", "Solar Roof Z", "Geothermal Plant P"],
    "Technology": ["Solar", "Wind", "Hydropower", "Solar", "Geothermal"],
    "Capacity (MW)": [150, 300, 200, 50, 100],  # Megawatts
    "Cost (Million $)": [200, 400, 350, 100, 250],  # Project cost
    "Location": ["California", "Texas", "Washington", "Nevada", "Idaho"],
    "Completion Year": [2023, 2024, 2022, 2025, 2023]
}

# Create the DataFrame
projects_df = pd.DataFrame(data)

# Add a new column for Cost per MW
projects_df["Cost per MW (Million $)"] = projects_df["Cost (Million $)"] / projects_df["Capacity (MW)"]

print(projects_df)


# In[ ]:


#create a pandas series for renewable enery sources

renewable_series = pd.Series(renewable_sources)

#print the series 
print("")


# In[1]:


# group by the technologuy and calculate total capacity for each type 

import pandas as pd 

# Data
data = {
    "Project": ["Solar Farm A", "Wind Turbine X", "Hydropower Y", "Solar Roof Z", "Geothermal Plant P"],
    "Technology": ["Solar", "Wind", "Hydropower", "Solar", "Geothermal"],
    "Capacity (MW)": [150, 300, 200, 50, 100],  # Megawatts
    "Cost (Million $)": [200, 400, 350, 100, 250],  # Project cost
    "Location": ["California", "Texas", "Washington", "Nevada", "Idaho"],
    "Completion Year": [2023, 2024, 2022, 2025, 2023]
}

# Create the DataFrame
projects_df = pd.DataFrame(data)

# Group by Technology and calculate total Capacity
grouped_capacity = projects_df.groupby("Technology")["Capacity (MW)"].sum()

print(grouped_capacity)


# In[3]:


#aggregate the total capacity and cost 
import pandas as pd 

# Data
data = {
    "Project": ["Solar Farm A", "Wind Turbine X", "Hydropower Y", "Solar Roof Z", "Geothermal Plant P"],
    "Technology": ["Solar", "Wind", "Hydropower", "Solar", "Geothermal"],
    "Capacity (MW)": [150, 300, 200, 50, 100],  # Megawatts
    "Cost (Million $)": [200, 400, 350, 100, 250],  # Project cost
    "Location": ["California", "Texas", "Washington", "Nevada", "Idaho"],
    "Completion Year": [2023, 2024, 2022, 2025, 2023]
}

# Create the DataFrame
projects_df = pd.DataFrame(data)

# Aggregate total capacity and cost
total_capacity = projects_df["Capacity (MW)"].sum()
total_cost = projects_df["Cost (Million $)"].sum()

print(f"Total Capacity (MW): {total_capacity}")
print(f"Total Cost (Million $): {total_cost}")


# In[47]:


#write a code to find the city with highest carbon footprint

climate_data = [
    {"city": "City A", "temperature": 25, "carbon_footprint": 500},
    {"city": "City B", "temperature": 30, "carbon_footprint": 350},
    {"city": "City C", "temperature": 22, "carbon_footprint": 600},
    {"city": "City D", "temperature": 15, "carbon_footprint": 200},
    {"city": "City E", "temperature": 28, "carbon_footprint": 450},
]
 #write the code to find the city with highest carbon footprint
highest_carbon_footprint=0
for city in climate_data:
  if city["carbon_footprint"]>highest_carbon_footprint:
    highest_carbon_footprint=city["carbon_footprint"]
    city_with_highest_carbon_footprint=city["city"]
print(f"\n city with the highest carbon footprint:")
print(f"City with highest carbon footprint:{city_with_highest_carbon_footprint} kg CO2")


# In[ ]:





# In[53]:


def calculate_carbon_footprint(energy_consumption, emission_factor):
 
    return energy_consumption * emission_factor

# Example usage
energy_consumption = 500  # in kWh
emission_factor = 0.5    # in kgCO2 per kWh

carbon_footprint = calculate_carbon_footprint(energy_consumption, emission_factor)
print(f"The carbon footprint is: {carbon_footprint} kgCO2")


# In[1]:


3+4

