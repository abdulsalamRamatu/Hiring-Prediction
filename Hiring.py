#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n


# In[29]:


df=pd.read_csv('C:\\Users\\Ramatu\\.jupyter\\py-master\\ML\\2_linear_reg_multivariate\\Exercise\\hiring1.csv')
df


# In[30]:


new_df=df.replace(['two','three','five', 'seven','ten','eleven'],[2,3,5,7,10,11])
new_df


# In[36]:


import math
median_test=math.floor(new_df.test_score.median())
median_test
new_df


# In[40]:


new_df1=new_df.fillna({'experience':0, 'test_score': median_test})
new_df1


# In[42]:


reg= linear_model.LinearRegression()
reg.fit(new_df1[['experience', 'test_score', 'interview_score']], new_df1.salary)


# In[44]:


reg.coef_


# In[45]:


reg.intercept_


# In[46]:


reg.predict([[2,9,6]])


# In[47]:


reg.predict([[12,10,10]])

