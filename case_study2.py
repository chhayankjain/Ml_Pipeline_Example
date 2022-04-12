#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[56]:


import pandas as pd
df = pd.read_csv("casestudy.csv")


# In[16]:


df.shape





df[df['year']==2015].customer_email


# In[36]:


data2016 = df[df['year'] == 2016]


# In[37]:


len(df[df['year']==2015]), len(df[df['year']==2016]), len(df[df['year']==2017])


# In[38]:


#new customer
data2016[~data2016.customer_email.isin(df[df['year']==2015].customer_email)].net_revenue.sum()


# In[39]:


#current year revenue
df.groupby('year').net_revenue.sum()


# In[40]:


df2015 = df[df.year == 2015]
df2016 = df[df.year == 2016]
df2017 = df[df.year == 2017]


# In[41]:


#new customer revenue
df2016[df2016.customer_email.isin(df2015.customer_email)].net_revenue.sum() - df2015[df2015.customer_email.isin(df2016.customer_email)].net_revenue.sum()


# In[55]:


#new customer revenue
df2017[df2017.customer_email.isin(df2016.customer_email)].net_revenue.sum() - df2016[df2016.customer_email.isin(df2017.customer_email)].net_revenue.sum()


# In[42]:


#prior year customer revenue 
df2016[df2016.customer_email.isin(df2015.customer_email)].net_revenue.sum()


# In[43]:


#prior year customer revenue 
df2015[df2015.customer_email.isin(df2016.customer_email)].net_revenue.sum()


# In[44]:


#attrition
df2015[~df2015.customer_email.isin(df2016.customer_email)].net_revenue.sum()


# In[45]:


df2016[~df2016.customer_email.isin(df2017.customer_email)].net_revenue.sum()


# In[46]:


#new customer revenue
df2017[df2017.customer_email.isin(df2016.customer_email)].net_revenue.sum() - df2016[df2016.customer_email.isin(df2017.customer_email)].net_revenue.sum()


# In[47]:


#current year customer revenue 
df2017[df2017.customer_email.isin(df2016.customer_email)].net_revenue.sum()


# In[48]:


#current year customer revenue 
df2016[df2016.customer_email.isin(df2017.customer_email)].net_revenue.sum()


# # Total revenue 

# In[49]:


#customer current year
len(df2015), len(df2016), len(df2017)


# In[50]:


#new customer
df2016[~df2016.customer_email.isin(df2015.customer_email)]


# In[51]:


df2017[~df2017.customer_email.isin(df2016.customer_email)]


# In[52]:


#lost customers
df2015[~df2015.customer_email.isin(df2016.customer_email)]


# In[54]:


df2016[~df2016.customer_email.isin(df2017.customer_email)]


# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:




