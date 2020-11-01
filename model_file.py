#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import preprocessing


# In[2]:


data = pd.read_csv('raw_salary_data.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data['experience'].fillna('zero',inplace=True)


# In[6]:


data['css'].fillna(data['css'].median(), inplace=True)
data['technical score'].fillna(data['technical score'].median(), inplace=True)


# In[7]:


data.isnull().sum()


# In[8]:


data.head()


# In[9]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()


# In[10]:


data['experience']=le.fit_transform(data['experience'])


# In[11]:


data.sample(10)


# In[12]:


X=data.drop('salary',axis=1)

y=data.pop('salary')


# In[13]:


X.shape


# In[14]:


y.shape


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.30,random_state=10)


# In[17]:


model_lr=LinearRegression()

model_lr.fit(X_train,y_train)
model_lr.score(X_train,y_train)


# In[18]:


prediction=model_lr.predict(X_test)


# In[19]:


pickle.dump(model_lr,open('model.pkl','wb'))


# In[20]:


model = pickle.load(open('model.pkl','rb'))
result=model.predict(X_test)

result


# In[21]:


data=[5,5,5,5,5,3,4,3,5,20]


# In[22]:


final=[np.array(data)]


# In[23]:


prediction1=model.predict(final)


# In[24]:


output=round(prediction1[0],2)


# In[25]:


output


# In[ ]:




