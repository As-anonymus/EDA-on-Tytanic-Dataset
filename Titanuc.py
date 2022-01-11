#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Dataset

# In[44]:


train = pd.read_csv(r'C:\Users\Aditya Singh\Downloads\titanic.csv')
df = train
train.head()


# In[5]:


## Statistics of dataset
train.describe()


# In[6]:


## datatype info
train.info()


# # Exploratory Data Analysis

# In[11]:


## categorical attributes
sns.countplot(train['survived'])


# In[12]:


sns.countplot(train['pclass'])


# In[13]:


sns.countplot(train['sex'])


# In[14]:


sns.countplot(train['sibsp'])


# In[15]:


sns.countplot(train['parch'])


# In[18]:


sns.countplot(train['embarked'])


# In[22]:


## numerical attributes 
sns.distplot(train['age'])


# In[23]:


sns.distplot(train['fare'])


# In[27]:


class_fare = train.pivot_table(index='pclass',values='fare')
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.xlabel('AvgFare')


# In[28]:


class_fare = train.pivot_table(index='pclass',values='fare',aggfunc=np.sum)
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.xlabel('TotalFare')


# In[60]:


sns.barplot(data=train,x='pclass',y='fare',hue='survived')


# In[45]:


#finding null
train.isnull().sum()-1#subtracted one to adjust the index


# In[42]:


train[train['pclass'].isnull()]


# In[46]:


#removing column
df = df.drop(columns=['cabin','boat','body','home.dest'],axis=1)


# In[47]:


df.head()


# In[49]:


#filling null for numerical
df['age'] = df['age'].fillna(df['age'].mean())
df['fare'] = df['fare'].fillna(df['fare'].mean())


# In[53]:


df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])


# # Log transform of uniformity

# In[54]:


sns.distplot(train['fare'])


# In[57]:


df['fare']=np.log(df['fare']+1)
sns.distplot(df['fare'])


# # Correlation

# In[58]:


corr=df.corr()


# In[59]:


plt.figure(figsize=(15,9))
sns.heatmap(corr,annot=True,cmap='coolwarm')


# In[61]:


df.head()


# In[62]:


#dropping more
df = df.drop(columns = ['name','ticket'],axis =1)
df.head()


# In[ ]:




