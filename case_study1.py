#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# to display all the columns
pd.pandas.set_option('display.max_columns',None)


# In[3]:


df = pd.read_csv('loans_full_schema.csv')
df.head()


# In[64]:


df.shape


# In[4]:


#Dropping emp title as it has 4000+ category 
df.drop('emp_title', axis = 1, inplace = True)


# # Missing Fields

# In[5]:


#make the list of features which has missing values
na_features = [features for features in df.columns if df[features].isnull().sum()>1]


#percentage of mission values
for feature in na_features:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values')


# # Relation between missing values and interest_rate

# In[6]:


for feature in na_features:
    data = df.copy()
    
    # adding a new feature just to indicate if the observation was missing
    data[feature] = np.where(data[feature].isnull(),1,0)

    #median  where the the info was missing
    data.groupby(feature)['interest_rate'].median().plot.bar()
    plt.title(feature)
    plt.show()


# In[7]:


#the above graphs show how interst rate differ if there are missing value in particluar feature


# # Listing out all numerical features

# In[8]:


#listing all the numberical features
num_features = [feature for feature in df.columns if df[feature].dtype != 'O']

print('Total Numerical features are: ', len(num_features))


# In[65]:


num_features


# In[9]:


df[num_features].head()


# # Listing out temporal features

# In[10]:


tempo_var = ['earliest_credit_line', 'issue_month']
tempo_var

for feature in tempo_var:
    print(feature, df[feature].unique())
    


# In[11]:


# relation between interest rate and earliest_credit_line
df.groupby('earliest_credit_line')['interest_rate'].median().plot()
plt.xlabel('earliest_credit_line')
plt.ylabel('interest_rate')
plt.title("earliest_credit_line vs interest_rate")


# In[12]:


# We can see that for customers after 1980 interest rate is somewhat incresing 


# # Seperating Continious and Discrete Features

# ## Discrete Features

# In[13]:


#considering a feature is discrete if it has less than 10 unique values 

dis_features = [feature for feature in num_features if len(df[feature].unique())<10 and feature not in tempo_var]
print("Discrete Variables Count: {}".format(len(dis_features)))


# In[14]:


dis_features


# In[15]:


df[dis_features].head()


# # Relationship between discreate features and interest rates

# In[16]:


for feature in dis_features:
    data=df.copy()
    data.groupby(feature)['interest_rate'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('interest_rate')
    plt.title(feature)
    plt.show()


# ## Continuous Features

# In[17]:


#Listing out continious variables
con_feature=[feature for feature in num_features if feature not in dis_features]
print("Continuous feature Count {}".format(len(con_feature)))


# In[18]:


con_feature


# # Relationship between discreate features and interest rates

# In[19]:


for feature in con_feature:
    data= df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[20]:


# If we want to apply regression at a later stage it's best to convert the continious feature to normal distribution


# In[21]:


for feature in con_feature:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        #data['interest_rate'] = np.log(data['interest_rate'])
        
        plt.scatter(data[feature], data['interest_rate'])
        plt.xlabel(feature)
        plt.ylabel('interest_rate')
        plt.title(feature)
        plt.show()
        
        


# # Checking for outliers

# In[22]:


for feature in con_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# # Categorical Features

# In[23]:


#listing out categorocal features
cat_features=[feature for feature in df.columns if df[feature].dtypes=='O']
cat_features


# In[24]:


df[cat_features].head()


# In[25]:


#checking for number of category
for feature in cat_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))


# In[26]:


#removing the 1st catergory i.e emp_title as there are 4724 different types
#cat_features.pop(0)


# In[27]:


cat_features


# # Relation betweeen categorical variable and interest rate

# In[28]:


for feature in cat_features:
    data=df.copy()
    data.groupby(feature)['interest_rate'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('interest_rate')
    plt.title(feature)
    plt.show()


# # Handeling Missing categorical value

# In[29]:


na_features_cat=[feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes=='O']

for feature in na_features_cat:
    print("{}: {}% missing values".format(feature,np.round(df[feature].isnull().mean(),4)))


# In[30]:


## Replacing missing value with a new label
def replace_cat_feature(df,na_features_cat):
    data=df.copy()
    data[na_features_cat]=data[na_features_cat].fillna('Missing')
    return data

dataset=replace_cat_feature(df,na_features_cat)

dataset[na_features_cat].isnull().sum()


# # Handeling Missing numerical value 

# In[31]:


na_features_num=[feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes!='O']

for feature in na_features_num:
    print("{}: {}% missing values".format(feature,np.round(df[feature].isnull().mean(),4)))


# In[32]:


## Replacing the numerical Missing Values with median as there are some outliers present

for feature in na_features_num:
    median_value=df[feature].median()
    
    ## create a new feature to capture nan values
    df[feature+'nan']=np.where(df[feature].isnull(),1,0)
    df[feature].fillna(median_value,inplace=True)
    
df[na_features_num].isnull().sum()


# In[33]:


df.head()


# In[ ]:





# # handeling rare category 

# In[34]:


#category with less than 1 percent apperace is considered rare catregory 
for feature in cat_features:
    temp=dataset.groupby(feature)['interest_rate'].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    df[feature]=np.where(df[feature].isin(temp_df),df[feature],'Rare_var')


# In[35]:


df.head(50)


# In[36]:


#Handeling categorical variable and converting it to numberical
for feature in cat_features:
    labels_ordered=df.groupby([feature])['interest_rate'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)


# In[37]:


df.head()


# # Feature Scaling

# In[38]:


scaling_feature=[feature for feature in df.columns if feature not in ['interest_rate'] ]
len(scaling_feature)


# In[39]:


feature_scale=[feature for feature in df.columns if feature not in ['interest_rate']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[feature_scale])


# In[40]:


scaler.transform(df[feature_scale])


# In[41]:


# transform the dataset, and add on the interest rate
data = pd.concat([df[['interest_rate']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(df[feature_scale]), columns=feature_scale)],
                    axis=1)


# # Modeling

# In[42]:


df_modelling = data.copy()


# In[43]:


df_modelling.drop('interest_rate', axis = 1, inplace = True)


# In[44]:


X = df_modelling.copy()


# In[45]:


y = data['interest_rate']


# # Feature Selection

# In[46]:


#Using Lasso regression to find the right features
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X, y)


# In[47]:


feature_sel_model.get_support()


# In[48]:


#total selected features

#list of the selected features
selected_feat = X.columns[(feature_sel_model.get_support())]

print('total features: {}'.format((X.shape[1])))
print('selected features: {}'.format(len(selected_feat)))


# In[49]:


X = X[selected_feat]


# In[61]:


selected_feat


# In[50]:


X


# In[51]:


y


# In[52]:


from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error


# In[53]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Ridge Regression Modelling 

# In[54]:


# instantiate Regression object
ridge = RidgeCV(cv=10)

# fit or train the linear regression model on the training set and store parameters
ridge.fit(x_train, y_train)

# show the alpha parameter used in final ridgeCV model
ridge.alpha_

# show the coefficients of each variable
# ridge.coef_


# In[55]:


y_pred = ridge.predict(x_test)
print(y_pred)


# In[56]:


from sklearn.metrics import r2_score


# # R2 score

# In[57]:


r2_score(y_test, y_pred)


# # Decision Tree Regressor

# In[58]:


#import the regressor
from sklearn.tree import DecisionTreeRegressor 
  
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
regressor.fit(x_train, y_train)


# In[59]:


y_pred_d = regressor.predict(x_test)
print(y_pred_d)


# # R2 Score

# In[60]:


r2_score(y_test, y_pred_d)


# In[62]:


sns.scatterplot(y_test, y_pred_d)


# In[63]:


sns.scatterplot(y_test, y_pred_d, marker = "+")


# In[ ]:


#We can se most of our predictios are very accurate except for some prediction where interest rate is high

