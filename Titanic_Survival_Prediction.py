#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


titanic = pd.read_csv("E:\Internship Task\AfameTechnologies\Titanic-Dataset.csv")


# In[3]:


titanic.head()


# In[4]:


titanic.tail()


# In[5]:


titanic.shape


# In[6]:


titanic.columns


# #Data Preprocessing and Data Cleaning
# 
# 
# 

# In[7]:


# Checking for data types
titanic.dtypes


# In[8]:


# checking for duplicated values
titanic.duplicated().sum()


# In[9]:


# checking for null values
nv = titanic.isna().sum().sort_values(ascending=False)
nv = nv[nv>0]
nv


# In[10]:


titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)


# In[11]:


titanic.info()


# In[12]:


# Cheecking what percentage column contain missing values
titanic.isnull().sum().sort_values(ascending=False)*100/len(titanic)


# In[13]:


# Since Cabin Column has more than 75 % null values .So , we will drop this column
titanic.drop(columns = 'Cabin', axis = 1, inplace = True)
titanic.columns


# In[14]:


# Filling Null Values in Age column with mean values of age column
titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)

# filling null values in Embarked Column with mode values of embarked column
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)

titanic.isna().sum()


# In[15]:


# Finding no. of unique values in each column of dataset
titanic[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']].nunique().sort_values()


# In[16]:


#how many survived?
titanic['Survived'].value_counts()


# # Dropping Some Unnecessary Columns
# 
# 
# >
# 
# 
# 
# 

# In[17]:


titanic.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic.columns


# In[18]:


titanic.info()


# In[19]:


# showing info. about numerical columns
titanic.describe()


# # **Data Visualization**

# In[20]:


d1 = titanic['Sex'].value_counts()
d1


# In[21]:


# Plotting Count plot for sex column
sns.countplot(x=titanic['Sex'])
plt.show()


# In[22]:


# Plotting Percantage Distribution of Sex Column
plt.figure(figsize=(5,5))
plt.pie(d1.values,labels=d1.index,autopct='%.2f%%')
plt.legend()
plt.show()


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt

# Convert 'Survived' column to string type
titanic['Survived'] = titanic['Survived'].astype(str)

# Create the count plot
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Survived', data=titanic)
plt.title('Distribution of Sex with respect to Survival')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', loc='best')
plt.show()


# In[27]:


# Showing Distribution of Embarked Sex wise
sns.countplot(x=titanic['Embarked'],hue=titanic['Sex'])
plt.show()


# In[28]:


# Plotting CountPlot for Pclass Column
sns.countplot(x=titanic['Pclass'])
plt.show()


# In[29]:


# Showing Distribution of Pclass Sex wise
sns.countplot(x=titanic['Pclass'],hue=titanic['Sex'])
plt.show()


# In[30]:


# Age Distribution
sns.kdeplot(x=titanic['Age'])
plt.show()


# In[31]:


# Plotting CountPlot for Survived Column
print(titanic['Survived'].value_counts())
sns.countplot(x=titanic['Survived'])
plt.show()


# In[32]:


# Showing Distribution of Parch Survived Wise
sns.countplot(x=titanic['Parch'],hue=titanic['Survived'])
plt.show()


# In[33]:


# Showing Distribution of SibSp Survived Wise
sns.countplot(x=titanic['SibSp'],hue=titanic['Survived'])
plt.show()


# In[34]:


# Showing Distribution of Embarked Survived wise
sns.countplot(x=titanic['Embarked'],hue=titanic['Survived'])
plt.show()


# In[35]:


# Showinf Distribution of Age Survived Wise
sns.kdeplot(x=titanic['Age'],hue=titanic['Survived'])
plt.show()


# In[37]:


# Plotting Histplot for Dataset
titanic.hist(figsize=(10,10))
plt.show()


# In[38]:


# Plotting pairplot
sns.pairplot(titanic)
plt.show()


# In[39]:


# converting categorical Columns

titanic.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[40]:


#Checking the target variable

titanic['Survived'].value_counts()


# In[41]:


sns.countplot(x=titanic['Survived'])
plt.show()


# In[42]:


titanic.head()


# # Data **Modelling**

# In[43]:


# importing libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[44]:


#Selecting the independent and dependent Features
cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = titanic[cols]
y = titanic['Survived']
print(x.shape)
print(y.shape)
print(type(x))  # DataFrame
print(type(y))  # Series


# In[45]:


x.head()


# In[46]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[47]:


y.head()


# In[48]:


#Creating Functions to compute Confusion Matrix, Classification Report and to generate Training and the Testing Score(Accuracy)


def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(ytest,ypred))

def mscore(model):
    print('Training Score',model.score(x_train,y_train))  # Training Accuracy
    print('Testing Score',model.score(x_test,y_test))     # Testing Accuracy


# 
# # Logistic Regression

# In[49]:


# Building the logistic Regression Model
lr = LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)
mscore(lr)
# Generating Prediction
ypred_lr = lr.predict(x_test)
print(ypred_lr)

# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_lr)
acc_lr = accuracy_score(y_test,ypred_lr)
print('Accuracy Score',acc_lr)


# 
# # KNN Classifier Model

# In[50]:


# Building the knnClassifier Model
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)
# Computing Training and Testing score
mscore(knn)
# Generating Prediction
ypred_knn = knn.predict(x_test)
print(ypred_knn)

# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_knn)
acc_knn = accuracy_score(y_test,ypred_knn)
print('Accuracy Score',acc_knn)


# # SVC model

# In[51]:


# Building Support Vector Classifier Model
svc = SVC(C=1.0)
svc.fit(x_train, y_train)
# Computing Training and Testing score
mscore(svc)
# Generating Prediction
ypred_svc = svc.predict(x_test)
print(ypred_svc)

# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_svc)
acc_svc = accuracy_score(y_test,ypred_svc)
print('Accuracy Score',acc_svc)


# 
# # Random Forest Classifier

# In[52]:


# Building the RandomForest Classifier Model
rfc=RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=5,max_depth=10)
rfc.fit(x_train,y_train)
# Computing Training and Testing score
mscore(rfc)
# Generating Prediction
ypred_rfc = rfc.predict(x_test)
print(ypred_rfc)

# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_rfc)
acc_rfc = accuracy_score(y_test,ypred_rfc)
print('Accuracy Score',acc_rfc)


# # DecisionTree **Classifier**

# In[53]:


# Building the DecisionTree Classifier Model
dt = DecisionTreeClassifier(max_depth=5,criterion='entropy',min_samples_split=10)
dt.fit(x_train, y_train)
# Computing Training and Testing score
mscore(dt)
# Generating Prediction
ypred_dt = dt.predict(x_test)
print(ypred_dt)
# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_dt)
acc_dt = accuracy_score(y_test,ypred_dt)
print('Accuracy Score',acc_dt)


# 
# 
# # Adaboost Classifier

# In[54]:


# Builing the Adaboost model
ada_boost  = AdaBoostClassifier(n_estimators=80)
ada_boost.fit(x_train,y_train)
mscore(ada_boost)
# Generating the predictions
ypred_ada_boost = ada_boost.predict(x_test)
# Evaluate the model - confusion matrix, classification Report, Accuracy Score
cls_eval(y_test,ypred_ada_boost)
acc_adab = accuracy_score(y_test,ypred_ada_boost)
print('Accuracy Score',acc_adab)


# In[55]:


#creating data frame

models = pd.DataFrame({
    'Model': ['Logistic Regression','knn','SVC','Random Forest Classifier','Decision Tree Classifier','Ada Boost Classifier'],
    'Score': [acc_lr,acc_knn,acc_svc,acc_rfc,acc_dt,acc_adab]})

models.sort_values(by = 'Score', ascending = False)


# In[60]:


colors = ["blue", "green", "pink", "cyan","black","purple"]

sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette=colors )
plt.show()


# # SINCE DECISION TREE IS BEST MODEL THAT PREDICTS TITANIC SURVIAL WITH MORE ACCURACY

# In[ ]:




