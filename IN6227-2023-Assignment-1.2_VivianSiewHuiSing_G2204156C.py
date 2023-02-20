#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Fetching Data
columns = ["age", "workclass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
train = pd.read_csv('~/Downloads/adult.data.csv', names=columns, sep=' *, *', engine='python', na_values='?')
test = pd.read_csv('~/Downloads/adult.test.csv', names=columns, sep=' *, *', engine='python', skiprows=1, na_values='?')

train.head()


# In[3]:


test.head()


# In[4]:


#Dealing Missing Values
train[train=='?']=np.nan
print(train.isnull().sum(),'\n')
print('Dimensions:',train.shape)


# In[5]:


train = train.dropna(axis=0)
train.info()


# In[6]:


test[test=='?']=np.nan
print(test.isnull().sum(),'\n')
print('Dimensions:',test.shape)


# In[7]:


test = test.dropna(axis=0)
test.info()


# In[8]:


#EDA
cat_attributes = train.select_dtypes(include=['object'])
cat_attributes.describe()


# In[9]:


num_attributes = train.select_dtypes(include=['int'])
num_attributes.describe()


# In[10]:


def plot(column):
    if train[column].dtype != 'int64':
        f, axes = plt.subplots(1,1,figsize=(15,5))
        sns.countplot(x=column, hue='income', data = train)
        plt.xticks(rotation=90)
        plt.suptitle(column,fontsize=20)
        plt.show()
    else:
        g = sns.FacetGrid(train, row="income", margin_titles=True, aspect=4, height=3)
        g.map(plt.hist,column,bins=100)
        plt.show()
    plt.show()


# In[11]:


plot('age')
plot('fnlwgt')
plot('education-num')
plot('capital-gain')
plot('capital-loss')
plot('hours-per-week')


# In[12]:


plot('workclass')
plot('education')
plot('marital-status')
plot('occupation')
plot('relationship')
plot('race')
plot('sex')
plot('native-country')
plot('income')


# In[13]:


train["income"]=train["income"].map({"<=50K":0,">50K":1})
test["income"]=test["income"].map({"<=50K.":0,">50K.":1})

import scipy.stats as stats
a=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','income']
for i in a:
    print(i,':',stats.pointbiserialr(train['income'],train[i])[0])


# In[14]:


#Feature Encoding
from sklearn.preprocessing import LabelEncoder
for col in train.columns:
    if train[col].dtypes == 'object':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        
for col in test.columns:
    if test[col].dtypes == 'object':
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].astype(str))


# In[15]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

count_incoding_list = dict(train['native-country'].value_counts())
train['native-country'] = train['native-country'].map(count_incoding_list)
train[['native-country']] = mms.fit_transform(train[['native-country']])
train[numerical] = mms.fit_transform(train[numerical])

count_incoding_list = dict(test['native-country'].value_counts())
test['native-country'] = test['native-country'].map(count_incoding_list)
test[['native-country']] = mms.fit_transform(test[['native-country']])
test[numerical] = mms.fit_transform(test[numerical])


# In[16]:


train_X=train.drop(["fnlwgt","education"],axis=1)
test_X=test.drop(["fnlwgt","education"],axis=1)

train_y=train["income"]
test_y=test["income"]

del train_X["income"]
del test_X["income"]


# In[17]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[18]:


#kNN
from sklearn.neighbors import KNeighborsClassifier

error=[]
for i in range(15,35,1):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(train_X,train_y)
    knn_pred = knn_model.predict(test_X)
    error.append(np.mean(knn_pred != test_y))
    
    print('\nk = ', i )
    print('Train Accuracy Score = ', round(knn_model.score(train_X, train_y) * 100, 4),"%")
    print("Test Accuracy Score:", round(accuracy_score(test_y, knn_pred) * 100, 4),"%")
    print("F1 Score: ", round(f1_score(test_y, knn_pred) * 100,4),"%")
    print("MSE: ", round(mean_squared_error(test_y, knn_pred) * 100,4),"%")
    
    print(confusion_matrix(test_y,knn_pred))
    print(classification_report(test_y,knn_pred))


# In[19]:


plt.figure(figsize=(12, 6))
plt.plot(range(15,35,1), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Squared Error')


# In[20]:


k=24

metric_accuracy_training = {}
metric_accuracy_testing = {}

dist_calc = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

for i in dist_calc:
    
    model = KNeighborsClassifier(k, metric = i) 
    model.fit(train_X, train_y)
    metric_accuracy_training[i] = model.score(train_X, train_y)
    metric_accuracy_testing[i] = model.score(test_X, test_y) 
    
    print('\nCalculation : ', i)
    print('Training accuracy :  ', knn_model.score(train_X, train_y))
    print('Testing accuracy : ', knn_model.score(test_X, test_y) )


# In[21]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state = 42)
dt_model.fit(train_X, train_y)
dt_pred = dt_model.predict(test_X)
print('Train accuracy score:', round(dt_model.score(train_X, train_y) * 100, 4))
print('Test Accuracy score:', round(accuracy_score(test_y, dt_pred) * 100, 4))
print("F1 Score: ", round(f1_score(test_y,dt_pred) * 100,4))
print("MSE: ", round(mean_squared_error(test_y,dt_pred) * 100,4))


# In[22]:


from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100,150],
    'criterion': ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params,cv=5, n_jobs=1, verbose=1, scoring = "accuracy")
grid_search.fit(train_X, train_y)


# In[23]:


dt_score = pd.DataFrame(grid_search.cv_results_)
dt_score


# In[24]:


grid_search.best_params_


# In[25]:


grid_search.best_score_


# In[26]:


dt_tuned = DecisionTreeClassifier(criterion="gini",max_depth=20,min_samples_leaf=50)
dt_tuned.fit(train_X,train_y)


# In[27]:


dt_tuned.score(test_X,test_y)


# In[28]:


#cross validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['kNN','Decision Tree']
models=[KNeighborsClassifier(n_neighbors=10),DecisionTreeClassifier()]
for i in models:
    model = i
    cv_result = cross_val_score(model,train_X,train_y, cv = kfold, scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
models_dataframe


# In[ ]:




