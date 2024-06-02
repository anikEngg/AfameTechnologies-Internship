#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("E:\Internship Task\AfameTechnologies\HR Data.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# # Data Preprocessing and Data Cleaning

# In[8]:


#checking for null values
df.isna().sum()


# In[9]:


df['DailyRate'] = df['DailyRate'].astype(int)
mean_daily_rate = df['DailyRate'].mean()
df['DailyRate'].fillna(mean_daily_rate, inplace=True)


# In[10]:


print(df.head())


# # Removing unnecessary **columns**

# In[11]:


columns_to_remove = ['EmployeeCount', 'StandardHours']
df_cleaned = df.drop(columns=columns_to_remove)
print(df_cleaned.head())


# # Eliminate the dataset's NaN values

# In[12]:


df_cleaned = df.dropna()
print(df_cleaned.head())


# # Eliminating redundant entries

# In[13]:


df.drop_duplicates()
df.dropna()


# # Giving the columns new names

# In[14]:


df.rename(columns={'DistanceFromHome': 'distance'}, inplace=True)
print(df['distance'])


# # Other process : Visualization

# In[15]:


sns.histplot(df['MonthlyIncome'],kde=True,bins=50)


# In[16]:


sns.histplot(data = df,x='Age',kde =True,hue='Attrition',)


# In[17]:


sns.histplot(df['Education'])
plt.xticks([1,2,3,4,5]);


# In[18]:


#job satisfaction impact

plt.figure(figsize=(10, 5))
sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
plt.show();
pd.crosstab(df.JobSatisfaction, df.Attrition,
            normalize='index')


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrame from a CSV file
# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv("E:\Internship Task\AfameTechnologies\HR Data.csv")

# Ensure 'Education' and 'Attrition' columns are of type string or category
df['Education'] = df['Education'].astype(str)
df['Attrition'] = df['Attrition'].astype(str)

# Check unique values to inspect the data
print("Unique values in 'Education':", df['Education'].unique())
print("Unique values in 'Attrition':", df['Attrition'].unique())

# Create a count plot
plt.figure(figsize=(12, 8))
sns.countplot(x='Attrition', hue='Education', data=df)
plt.legend(loc='best')
plt.title('Count Plot of Attrition by Education')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.show()

# Create a cross-tabulation
education_attrition_crosstab = pd.crosstab(df['Education'], df['Attrition'], normalize='index')
print(education_attrition_crosstab)


# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(data=df,y='JobRole')
plt.title("Job Role")
plt.show()


# In[22]:


plt.figure(figsize=(10,5))
sns.countplot(data=df,x="JobRole",hue="Attrition")
plt.title("Attrition by Job Role")
plt.xticks(rotation=90)
plt.show()


# In[23]:


# Department wise age

plt.figure(figsize=(10,5))
sns.boxplot(data=df,x="Department",y="Age")
plt.title("Age Distribution By Department")

plt.show()

