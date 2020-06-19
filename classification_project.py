#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('train.csv')


# In[3]:


data.head()


# In[4]:


a=data.iloc[3,1:].values


# In[5]:


a=a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[6]:


df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


# In[ ]:





# In[7]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)


# In[8]:


x_train.head()


# In[9]:


y_train.head()


# In[10]:


rf=RandomForestClassifier(n_estimators=100)


# In[11]:


rf.fit(x_train,y_train)


# In[12]:


pred=rf.predict(x_test)


# In[13]:


pred


# In[14]:


s=y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1


# In[15]:


count


# In[16]:


len(pred)


# In[17]:


8077/8400


# In[ ]:




