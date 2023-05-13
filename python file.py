#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling
# ## Data gathering

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[2]:


df = pd.read_csv("E:\مشاريع للزمن\Data_science_project\Banknote-authentication-dataset-.csv")


# ## Data Assessing

# In[3]:


df.describe()


# In[4]:


df.info()


# ## Data Cleaning

# In[5]:


df.isna().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df.drop_duplicates(inplace=True)


# # Data Exploration

# In[8]:


plt.scatter(df.V1, df.V2)


# In[9]:


kmeans = KMeans(n_clusters=2).fit(df)
plt.scatter(df.V1, df.V2, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 2000, alpha=.5)
plt.xlabel("V1")
plt.ylabel("V2")
plt.title("Clustering data into two clusters")

