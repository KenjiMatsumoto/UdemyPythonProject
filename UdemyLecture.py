
# coding: utf-8

# # データの読み込み

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv('housing.csv')


# In[4]:


df.head(3)


# In[5]:


len(df)


# In[6]:


df.describe()

