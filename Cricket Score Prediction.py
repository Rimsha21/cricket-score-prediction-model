
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


import pickle


# In[18]:


df = pd.read_csv('ipl.csv')


# In[19]:


df.head()


# In[20]:


df['bat_team'].unique()


# In[21]:


df = df[df['overs']>=5.0]


# In[22]:


df.head()


# In[23]:


print(df['bowl_team'].unique())


# In[24]:


from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[25]:


encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# In[51]:


encoded_df.columns


# In[27]:


encoded_df.columns


# In[28]:


X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]


# In[29]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[30]:


X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)


# In[32]:


X_train.drop(labels='venue',axis=True, inplace=True)
X_test.drop(labels='venue', axis=True, inplace=True)


# In[34]:


X_train.drop(labels='batsman',axis=True, inplace=True)
X_test.drop(labels='batsman', axis=True, inplace=True)


# In[35]:


X_train.drop(labels='bowler',axis=True, inplace=True)
X_test.drop(labels='bowler', axis=True, inplace=True)


# In[50]:


encoded_df.head()


# In[36]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[37]:


filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# In[38]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


# In[39]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[41]:


prediction = ridge_regressor.predict(X_test)


# In[42]:


prediction


# In[43]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[44]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

