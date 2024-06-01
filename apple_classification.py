#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Required libraries 


# In[120]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


#import data from local system 


# In[4]:


data=pd.read_csv('C:/Users/Hitesh/Downloads/apple_quality.csv')
data.head()


# In[5]:


# checikng how much data we have 


# In[6]:


data.shape


# In[7]:


# checking is their any value is null or not


# In[8]:


data.isna().sum()


# In[9]:


# Now we are seeing which is null and if the number of row is null then we have to replace null values with values that can help us to predict model correctly


# In[10]:


data[data['Size'].isna()]


# In[11]:


# as number of row is just 1 so we can drop it and it won't effect the data that much


# In[12]:


data=data.dropna()


# In[13]:


data.describe()


# In[14]:


data.isna().sum()


# In[15]:


#Now we are dividing the data into features and target , that will help us to create the model 

# Features are just the input data that goes to model and target is just a ouput comes from our model


# In[ ]:


#So now we are droping the id as A_id don't help in the model as id don't specifiy any specific information regarding the apple quality .


# In[16]:


feature=data.drop(['A_id','Quality'],axis=1)
feature.head()


# In[18]:


# Now I just want to see how many apple is good and how many are bad


# In[19]:


data['Quality'].value_counts()


# In[20]:


# And we also have to convert the object data type as int as our model only understand numberical thing


# In[21]:


target=data['Quality']
target = data['Quality'].replace({'good': 1, 'bad': 0}, inplace=False)


# In[22]:


target.value_counts()


# In[23]:


data['Quality'].replace({'good': 1, 'bad': 0}, inplace=True)


# In[25]:


data.drop('A_id',axis=1,inplace=True)


# In[29]:


# train_test_split helps us to divide our data into train and test so we can train our model at particular data and test data on other dataset and see the model accuracy


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(feature,target,test_size=0.2,random_state=5)


# In[31]:


df = pd.DataFrame(X_train, columns=X_train.columns.tolist())
df.head()


# In[32]:


from pandas.plotting import scatter_matrix


# In[33]:


graph=scatter_matrix(df,c=y_train,figsize=(10,10),marker='o',s=60,alpha=.8,hist_kwds={'bins':20})


# As we can see from above plot all the data set are intercepting each other so that it will not able to separate each other so we  can't apply k-nn here

# ## The notebook trains multiple models and evaluates their performance, including K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, and Support Vector Machine (SVM).

# # Kneighborsclassifier

# In[34]:


from sklearn.neighbors  import KNeighborsClassifier
model1=KNeighborsClassifier(n_neighbors=1)


# In[35]:


model1.fit(X_train,y_train)


# In[36]:


print('Train:',model1.score(X_train,y_train))
print('Test:', model1.score(X_test,y_test))


# In[37]:


model1a=KNeighborsClassifier(n_neighbors=3)
model1a.fit(X_train,y_train)


# In[38]:


print('Train:',model1a.score(X_train,y_train))
print('Test:', model1a.score(X_test,y_test))


# In[39]:


model1b=KNeighborsClassifier(n_neighbors=10)
model1b.fit(X_train,y_train)


# In[40]:


print('Train:',model1b.score(X_train,y_train))
print('Test:', model1b.score(X_test,y_test))


# In[41]:


model1c=KNeighborsClassifier(n_neighbors=20)
model1c.fit(X_train,y_train)


# In[42]:


print('Train:',model1c.score(X_train,y_train))
print('Test:', model1c.score(X_test,y_test))   #50 underfitting


# # Linear Regression

# when we use regression model for classification the model learning is bad as you can see below

# In[43]:


from sklearn.linear_model import LinearRegression


# In[44]:


model2=LinearRegression().fit(X_train,y_train)


# In[45]:


print('Train:',model2.score(X_train,y_train))
print('Test:', model2.score(X_test,y_test)) 


# ## Ridge Model

# In[46]:


from sklearn.linear_model import Ridge


# In[47]:


model2a=Ridge().fit(X_train,y_train)


# In[48]:


print('Train:',model2a.score(X_train,y_train))
print('Test:', model2a.score(X_test,y_test)) 


# In[49]:


model2b=Ridge(alpha=10).fit(X_train,y_train) #control underfitting and overfitting


# In[50]:


print('Train:',model2b.score(X_train,y_train))
print('Test:', model2b.score(X_test,y_test)) 


# In[51]:


model2c=Ridge(alpha=0.01).fit(X_train,y_train) #control underfitting and overfitting


# In[52]:


print('Train:',model2c.score(X_train,y_train))
print('Test:', model2c.score(X_test,y_test)) 


# ##  Lasso
# 

# In[53]:


from sklearn.linear_model import Lasso


# In[54]:


model2d=Lasso().fit(X_train,y_train)


# In[55]:


print('Train:',model2d.score(X_train,y_train))
print('Test:', model2d.score(X_test,y_test)) 


# In[56]:


model2e=Lasso(alpha=0.01, max_iter=10000).fit(X_train,y_train)


# In[57]:


print('Train:',model2e.score(X_train,y_train))
print('Test:', model2e.score(X_test,y_test)) 


# In[58]:


model2f=Lasso(alpha=1, max_iter=10000000).fit(X_train,y_train)


# In[59]:


print('Train:',model2e.score(X_train,y_train))
print('Test:', model2e.score(X_test,y_test)) 


# In[ ]:





# # Linear Classifier

# ## Logistic Regression

# In[60]:


from sklearn.linear_model import LogisticRegression


# In[61]:


model3=LogisticRegression().fit(X_train,y_train)


# In[62]:


print('Train:',model3.score(X_train,y_train))
print('Test:', model3.score(X_test,y_test)) 


# In[63]:


model3a=LogisticRegression(C=1000).fit(X_train,y_train)


# In[64]:


print('Train:',model3a.score(X_train,y_train))
print('Test:', model3a.score(X_test,y_test)) 


# ## Linear Svc

# In[65]:


from sklearn.svm import LinearSVC


# In[66]:


model3b=LinearSVC().fit(X_train,y_train)


# In[67]:


print('Train:',model3b.score(X_train,y_train))
print('Test:', model3b.score(X_test,y_test)) 


# In[68]:


model3c=LinearSVC(C=50).fit(X_train,y_train)  #C=10 test desc  C=0.1 Same C=0.01  test Desc


# In[69]:


print('Train:',model3c.score(X_train,y_train))
print('Test:', model3c.score(X_test,y_test)) 


# # Naive Bayes Classifier

# In[70]:


from sklearn.naive_bayes import GaussianNB


# In[71]:


model4=GaussianNB().fit(X_train,y_train)


# In[72]:


print('Train:',model4.score(X_train,y_train))
print('Test:', model4.score(X_test,y_test)) 


# In[73]:


from sklearn.naive_bayes import BernoulliNB  #work best binary data


# In[74]:


model4a=BernoulliNB().fit(X_train,y_train)


# In[75]:


print('Train:',model4a.score(X_train,y_train))
print('Test:', model4a.score(X_test,y_test)) 


# In[76]:


from sklearn.naive_bayes import MultinomialNB


# In[77]:


## model4c=MultinomialNB().fit(X_train,y_train)    -> Give error it works beston Text type of data as it doesn't feature with negative values


# In[ ]:





# # Decision Tree

# In[78]:


from sklearn.tree import DecisionTreeClassifier


# In[79]:


model5=DecisionTreeClassifier().fit(X_train,y_train)


# In[80]:


print('Train:',model5.score(X_train,y_train))
print('Test:', model5.score(X_test,y_test)) 


# In[83]:


model5a=DecisionTreeClassifier(max_depth=10).fit(X_train,y_train)


# In[84]:


print('Train:',model5a.score(X_train,y_train))
print('Test:', model5a.score(X_test,y_test)) 


# In[85]:


depths = [1, 10, 100, 200]  # Renamed list to depths
for i in depths:  # Iterating through the depths list
    model5a = DecisionTreeClassifier(max_depth=i).fit(X_train, y_train)
    print('Train:', i,'depth: ', model5a.score(X_train, y_train))
    print('Test:', i ,'depth: ', model5a.score(X_test, y_test))


# In[86]:


from sklearn.tree import export_graphviz


# In[87]:


export_graphviz(model51,out_file="tree.dot",class_names=['good','bad'],impurity=False, filled=True)


# In[88]:


import graphviz


# In[89]:


with open("tree.dot") as f:
    dot_graph = f.read()
print(dot_graph)


# In[91]:


def plot_feat(model, X_train):
    n_features = X_train.shape[1]  # Get the number of features
    plt.barh(range(n_features), model.feature_importances_, align='center')  # Use model.feature_importances_
    plt.yticks(range(n_features), X_train.columns.tolist())  # Use range(n_features) instead of np.arange(n_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.show()


# In[92]:


plot_feat(model51,X_train)


# # Random Forest

# In[93]:


from sklearn.ensemble import RandomForestClassifier


# In[94]:


model6=RandomForestClassifier(n_estimators=5)


# In[95]:


model6.fit(X_train,y_train)


# In[96]:


print('Train:',model6.score(X_train,y_train))
print('Test:', model6.score(X_test,y_test)) 


# In[97]:


n_est=[10,20,50,100]
for i in n_est:
    model=RandomForestClassifier(n_estimators=i)
    model.fit(X_train,y_train)
    print('Train:',[i],model.score(X_train,y_train))
    print('Test:',[i], model.score(X_test,y_test)) 


# In[98]:


plot_feat(model6,X_train)


# # Gradient Boosting Machine

# In[99]:


from sklearn.ensemble import GradientBoostingClassifier


# In[100]:


model7=GradientBoostingClassifier(random_state=0)


# In[101]:


model7.fit(X_train,y_train)


# In[102]:


print('Train:',model7.score(X_train,y_train))
print('Test:', model7.score(X_test,y_test)) 


# In[103]:


n_est=[1,2,5,10]
for i in n_est:
    model=GradientBoostingClassifier(max_depth=i)
    model.fit(X_train,y_train)
    print('Train:',[i],model.score(X_train,y_train))
    print('Test:',[i], model.score(X_test,y_test)) 


# # KSVM
# 

# In[104]:


from sklearn.svm import LinearSVC


# In[105]:


model8=LinearSVC().fit(X_train,y_train)


# In[106]:


print('Train:',model8.score(X_train,y_train))
print('Test:', model8.score(X_test,y_test)) 


# In[107]:


model8a=LinearSVC(C=0.1).fit(X_train,y_train)


# In[108]:


print('Train:',model8a.score(X_train,y_train))
print('Test:', model8a.score(X_test,y_test)) 


# In[ ]:





# In[109]:


# Using Heat Map 


# In[110]:


import seaborn as sns


# In[111]:


plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, fmt='.3f', linewidths=0.4, cmap="cividis")
plt.show()


# In[112]:


feature.head()


# In[113]:


feature1=data.drop(['Crunchiness','Weight','Quality'],axis=1)
feature1.head()


# In[114]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1 , y_test1=train_test_split(feature1,target,random_state=10)


# In[115]:


model9=LinearSVC().fit(X_train1,y_train1)


# In[116]:


print('Train:',model9.score(X_train1,y_train1))
print('Test:', model9.score(X_test1,y_test1)) 


# In[117]:


model10=RandomForestClassifier(n_estimators=5).fit(X_train1,y_train1)


# In[118]:


print('Train:',model10.score(X_train1,y_train1))
print('Test:', model10.score(X_test1,y_test1)) 


# In[123]:





# In[ ]:




