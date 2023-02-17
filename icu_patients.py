#!/usr/bin/env python
# coding: utf-8

# In[270]:


import os
import pandas as pd
import numpy as np
import sqlite3 as db

import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score,recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#import the ICU_patients dataset stored
#in the folder ICU_patients

file =('ICU_patients/ICU.csv')

#ICU_patients dataset
data = pd.read_csv(file, index_col=0)

#displaying the dataset
data.head()


# In[271]:


#checking if there is any null values
data.info(verbose=True, show_counts=True)

#all the columns have the same number of entries as the number of rows
#hence all the columns have complete entries (no null entries)


# In[272]:


#data normalization

df1 = pd.read_csv(file, index_col=0)
df1.pop('AgeGroup')
df1.head()


# In[273]:


#separate agegroups (transitive dependency)
df2 = pd.read_csv(file,usecols = ['Age','AgeGroup'])

#unique age with the subsequent age groups
df2.drop_duplicates(keep="first", inplace=True)
df2.head()

#the number of entries
#df2.shape


# In[274]:


#data visualization
#sqlite normalized dataset

conn = db.connect('test.db',timeout = 20)
c = conn.cursor()

#table 1:patients
c.execute("DROP TABLE IF EXISTS icu")
c.execute("CREATE TABLE icu (ID int, Survive int, Age int,Sex int, Infection int, SysBP int, Pulse int, Emergency int)")
df1.to_sql('icu', conn, if_exists='append', index = False)
#c.execute("SELECT * FROM icu").fetchall()

#colnames = c.description
#for row in colnames:
    #print(row[0])


# In[275]:


#table 2: agegroups
c.execute("DROP TABLE IF EXISTS age")
c.execute("CREATE TABLE age (Age int,AgeGroup int)")
df2.to_sql('age', conn, if_exists='append', index = False)
#c.execute("SELECT * FROM age").fetchall()

#colnames = c.description
#for row in colnames:
    #print(row[0])


# In[276]:


#graph counting the number of patients who survived [==1] vs the number of patients who did not survive [==0]
plt.figure()
sns.countplot(data['Survive'])
plt.show()


# In[277]:


#split the data into test and train

#predictors - prediction features 
features = df1.drop(columns='Survive')

#outcome variable 
target = df1[['Survive']]

#train_test_split
x_train , x_test , y_train , y_test = train_test_split(features,target,test_size=0.2,random_state=2)


# In[278]:


#metrics dataframe
metrics = pd.DataFrame()


# In[279]:


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train,y_train.values.ravel())

#accuracy of the model
#train-model
y_train_pred = lr_model.predict(x_train)
train_lr_acc = accuracy_score( y_train.values.ravel(), y_train_pred)
print("Train Accuracy: ",train_lr_acc)
#test-model
y_test_pred = lr_model.predict(x_test)
test_lr_acc = accuracy_score( y_test.values.ravel(), y_test_pred)
print("Test Accuracy: ",test_lr_acc)


# In[ ]:


#Metrics of Logistic Regression Model


# In[280]:


#confusion matrix
cf = confusion_matrix(y_test.values.ravel(), y_test_pred)
print("Confusion Matrix:")
print(cf)

#metrics
precision = precision_score(y_test.values.ravel(), y_test_pred)
recall = recall_score(y_test.values.ravel(), y_test_pred)
f1 = f1_score(y_test.values.ravel(), y_test_pred)
roc = roc_auc_score(y_test.values.ravel(), y_test_pred)

lr = {"Train Accuracy":train_lr_acc,"Test_Accuracy":test_lr_acc,"Precision":precision, "Recall":recall, "F1":f1, "ROC": roc}
metrics = metrics.append(lr, ignore_index=True)
print("Precision: ",precision,
     "Recall: ",recall,
     "f1: ",f1,
     "ROC: ",roc)


# In[203]:


#Decision Tree


# In[281]:


df_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42, max_features = 'auto')

#train-model
df_model.fit(x_train, y_train)

#train-model
y_train_pred = df_model.predict(x_train)
train_df_acc = accuracy_score( y_train.values.ravel(), y_train_pred)
print("Train Accuracy: ",train_df_acc)

#test-model
y_test_pred = df_model.predict(x_test)

test_df_acc = accuracy_score( y_test.values.ravel(), y_test_pred)
print("Test Accuracy: ",test_df_acc)


# In[ ]:


#Metrics of Decision Tree Model


# In[282]:


#confusion matrix
cf = confusion_matrix(y_test.values.ravel(), y_test_pred)
print("Confusion Matrix:")
print(cf)

#metrics
precision_df = precision_score(y_test.values.ravel(), y_test_pred)
recall_df = recall_score(y_test.values.ravel(), y_test_pred)
f1_df = f1_score(y_test.values.ravel(), y_test_pred)
roc_df = roc_auc_score(y_test.values.ravel(), y_test_pred)
df = {"Train Accuracy":train_df_acc,"Test_Accuracy":test_df_acc,"Precision":precision_df, "Recall":recall_df, "F1":f1_df, "ROC": roc_df}
metrics = metrics.append(df, ignore_index=True)
print("Precision: ",precision_df,
     "Recall: ",recall_df,
     "f1: ",f1_df,
     "ROC: ",roc_df)


# In[283]:


#Random Forest

model_rf = RandomForestClassifier(n_estimators = 10,random_state=42)

#train-model
model_rf.fit(x_train, y_train.values.ravel())

#train-model
y_train_pred = model_rf.predict(x_train)
train_rf_acc = accuracy_score( y_train.values.ravel(), y_train_pred)
print("Train Accuracy: ",train_rf_acc)

#test-model
y_test_pred = model_rf.predict(x_test)

test_rf_acc = accuracy_score( y_test.values.ravel(), y_test_pred)
print("Test Accuracy: ",test_rf_acc)


# In[284]:


#Metrics of Random Forest Model

#confusion matrix
cf = confusion_matrix(y_test.values.ravel(), y_test_pred)
print("Confusion Matrix:")
print(cf)

#metrics
precision_rf = precision_score(y_test.values.ravel(), y_test_pred)
recall_rf = recall_score(y_test.values.ravel(), y_test_pred)
f1_rf = f1_score(y_test.values.ravel(), y_test_pred)
roc_rf = roc_auc_score(y_test.values.ravel(), y_test_pred)

rf = {"Train Accuracy":train_rf_acc,"Test_Accuracy":test_rf_acc,"Precision":precision_rf, "Recall":recall_rf, "F1":f1_rf, "ROC": roc_rf}
metrics = metrics.append(rf, ignore_index=True)

print("Precision: ",precision_rf,
     "Recall: ",recall_rf,
     "f1: ",f1_rf,
     "ROC: ",roc_rf)


# In[ ]:


metrics.index = ['Logistic Regression', 'Decsion Tree', 'Random Forest']
print(metrics)


# In[258]:


#Conclusion
#Decision tree and random forest have the best test accuracy.
#ROC (Receiver Operator Characteristic - that is the ability of the model to distinguish between the two classes),
# of the Random Forest Model is higher than the other two models in addition to precision

