#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install numpy


# In[10]:


# one d array
import numpy as np  # Import the NumPy library

# Create a one-dimensional array
one_d_array = np.array([1, 2, 3])
print(one_d_array)


# In[12]:


# create a 2-d array
import numpy as np
two_d_array = np.array([[1,2],[3,4]])
print(two_d_array)


# In[14]:


#use ndarray.shape
import numpy as np  # Import the NumPy library

# Create a one-dimensional array
one_d_array = np.array([1, 2, 3])

# Use ndarray.shape to get the shape of the array
print("Array:", one_d_array)
print("Shape of the array:", one_d_array.shape)


# In[16]:


# use ndarray.ndim
import numpy as np  # Import the NumPy library

# Create a one-dimensional array
one_d_array = np.array([1, 2, 3])

# Use ndarray.ndim to get the number of dimensions of the array
print("Array:", one_d_array)
print("Number of dimensions:", one_d_array.ndim)


# In[18]:


# use ndarray.size
import numpy as np  # Import the NumPy library

# Create a one-dimensional array
one_d_array = np.array([1, 2, 3])

# Use ndarray.size to get the total number of elements in the array
print("Array:", one_d_array)
print("Number of elements:", one_d_array.size)


# In[20]:


# use ndarray.dtype
import numpy as np  # Import the NumPy library

# Create a one-dimensional array with integers
one_d_array = np.array([1, 2, 3])

# Use ndarray.dtype to get the data type of the array elements
print("Array:", one_d_array)
print("Data type of the elements:", one_d_array.dtype)


# In[22]:


# use ndarray.itemsize

import numpy as np  # Import the NumPy library

# Create a one-dimensional array with integers
one_d_array = np.array([1, 2, 3])

# Use ndarray.itemsize to get the size (in bytes) of each element in the array
print("Array:", one_d_array)
print("Size of each element (in bytes):", one_d_array.itemsize)


# In[ ]:


# Numpy vs list 


# In[24]:


# diffrent ways of numpy 
import numpy as np

# Create a 2x3 array of ones
arr = np.ones((2, 3))
print(arr)

#np.zeros
import numpy as np

# Create a 3x2 array of zeros
arr = np.zeros((3, 2))
print(arr)

#np.random

import numpy as np

# Create a 2x3 array of random numbers between 0 and 1
arr = np.random.rand(2, 3)
print(arr)





# In[26]:


#crate 3 x3 an array where all elements are 10 use np.full 
import numpy as np

# Create a 3x3 array filled with 10
arr = np.full((3, 3), 10)

print(arr)


# In[28]:


#get the identity matrix where identity matris is 1
import numpy as np

# Create a 3x3 identity matrix
identity_matrix = np.eye(3)

print(identity_matrix)


# In[30]:


# create an array with 3 element between 5 to 10 using linespase
import numpy as np

# Create an array with 3 elements between 5 and 10
arr = np.linspace(5, 10, 3)

print(arr)


# In[32]:


# create an array with elements from 5 to 10 with stepsize 2
import numpy as np

# Create an array from 5 to 10 with a step size of 2
arr = np.arange(5, 11, 2)

print(arr)


# In[34]:


# arrange the values from 0 to 10 
import numpy as np

# Create an array with values from 0 to 10 (exclusive)
arr = np.arange(0, 11)

print(arr)


# In[36]:


#indexing and slicing 
# List example
lst = [10, 20, 30, 40, 50]

# Accessing an element using positive index
print(lst[0])  # First element
print(lst[2])  # Third element

# Accessing an element using negative index
print(lst[-1])  # Last element
print(lst[-2])  # Second-to-last element


# In[38]:


# List example
lst = [10, 20, 30, 40, 50]

# Slicing from index 1 to 3 (not including 3)
print(lst[1:3])  # Output: [20, 30]

# Slicing with step size of 2 (skipping every second element)
print(lst[::2])  # Output: [10, 30, 50]

# Slicing with negative indices
print(lst[-3:-1])  # Output: [30, 40]

# Slicing with step size of -1 (reversing the list)
print(lst[::-1])  # Output: [50, 40, 30, 20, 10]


# In[40]:


# String example
text = "Hello, World!"

# Indexing
print(text[0])   # 'H'
print(text[-1])  # '!'

# Slicing
print(text[0:5])  # 'Hello'
print(text[7:])   # 'World!'
print(text[::-1]) # '!dlroW ,olleH'


# In[42]:


import numpy as np

# 2D Numpy array example
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing a specific element
print(arr[1, 2])  # Element at 2nd row, 3rd column (6)

# Slicing a sub-array
print(arr[0:2, 1:3])  # Slicing rows 0 to 1 and columns 1 to 2


# In[46]:


# string palindrome 

string = "A man a plan a canal Panama"

# Remove spaces and convert to lowercase for case insensitivity
cleaned_string = string.replace(" ", "").lower()

# Check if the string is equal to its reverse
is_palindrome = cleaned_string == cleaned_string[::-1]

print("Palindrome" if is_palindrome else "Not a Palindrome")


# In[48]:


word = "radar"

# Reverse the word
is_palindrome = word == word[::-1]

# Output the result
print("Palindrome" if is_palindrome else "Not a Palindrome")


# In[50]:


# Array manupulation
# reshape 
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

reshaped_arr = arr.reshape(2, 3)

print("Original Array:")
print(arr)

print("Reshaped Array:")
print(reshaped_arr)


# In[52]:


# find the transpose of the matrix 
import numpy as np

# Define a 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose the matrix
transpose_matrix = matrix.T

print("Original Matrix:")
print(matrix)

print("Transposed Matrix:")
print(transpose_matrix)


# In[54]:


#flatten 
import numpy as np

# Define a 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Flatten the matrix
flattened_array = matrix.flatten()

print("Original Matrix:")
print(matrix)

print("Flattened Array:")
print(flattened_array)


# In[64]:


# find out mean median mode std and variance by numpy
import numpy as np
from scipy import stats

# Sample data
data = np.array([1, 2, 2, 3, 4, 5, 5, 5, 6, 7])

# Mean
mean = np.mean(data)

# Median
median = np.median(data)

# Mode
mode_result = stats.mode(data, keepdims=True)  # Ensure compatibility with newer versions of SciPy
mode_value = mode_result.mode[0]
mode_frequency = mode_result.count[0]

# Standard Deviation
std_dev = np.std(data)

# Variance
variance = np.var(data)

# Results
print("Data:", data)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode_value, "(Frequency:", mode_frequency, ")")
print("Standard Deviation:", std_dev)
print("Variance:", variance)



# In[62]:


#reshape 
import numpy as np

# Original array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Reshape the array into a 3x3 matrix
reshaped_arr = arr.reshape(3, 3)

print("Original Array:")
print(arr)

print("\nReshaped Array (3x3):")
print(reshaped_arr)

